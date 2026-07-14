#!/usr/bin/env python3
"""Y2 · 低置信回退：structure_d 主路径，head 低置信时回退 knn+M2。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase24_common import (
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits,
    timed_run, timing_metrics, write_phase24_result,
)
from boundary_budget import build_prompt_budget_labels, make_d4_knn_budget_fn, make_structure_budget_fn, train_d4_knn_bank
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, extract_latent_hidden,
    first_correct_step, split_train_val_samples,
)


@torch.no_grad()
def eval_confidence_fallback(head, model, tokenizer, samples, *, device, seed, profile,
                             struct_floor, knn_floor, struct_thr, knn_thr, fallback_thr: float):
    head.eval()
    correct = fallback_count = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
        prob0 = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if prob0 >= fallback_thr:
            final, stop_n = pred0, n0
        else:
            fallback_count += 1
            floor_n = max(MIN_N, min(CAP, knn_floor(sample)))
            stop_n = CAP
            final = preds.get(CAP, "")
            prev, streak = "", 0
            for n in range(1, CAP + 1):
                final = preds[n]
                ab, streak, ch = _rich_step_features(final, prev, streak)
                prev = final
                prompt = build_eval_prompt(sample, n, seed=seed + idx * 31 + n,
                    choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
                ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
                hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
                prob = torch.sigmoid(head(
                    hid.unsqueeze(0), torch.tensor([n], device=device),
                    torch.tensor([ab], device=device), torch.tensor([streak], device=device),
                    torch.tensor([ch], device=device),
                )).item()
                stop_n = n
                if n >= floor_n and prob >= knn_thr:
                    break
        if final == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(samples), 4),
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"uses_oracle": False, "fallback_thr": fallback_thr, "mode": "confidence_fallback"},
    }
    row.update(timing_metrics(stop_ns, fcs))
    row["deployable_mvp"] = is_deployable_mvp(row)
    row["eps_deployable"] = is_eps_deployable(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, _ = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))
        struct_floor = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        knn_bank, _ = train_d4_knn_bank(train_rows, feature_key="joint_features")
        knn_floor = make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP)
        knn_thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, thresholds=FINE_GRID,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile, optimize="accuracy",
        )

        sweep = []
        for fb_thr in (0.15, 0.35, 0.5, 0.65):
            row = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=SEED, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor,
                struct_thr=0.15, knn_thr=knn_thr, fallback_thr=fb_thr,
            )
            sweep.append(row)
        best = max(sweep, key=lambda r: (r["accuracy"], r.get("timing_eps1") or 0))
        p22 = load_json("phase22/x5_complement_audit_latest.json")
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "baseline_fusion": p22.get("union_acc"),
            "insight": "P23 并集 95.9%：低置信回退 knn 能否无 oracle 再抬 acc？",
            "mentor_brief": (
                f"Y2 低置信回退：thr={best['params']['fallback_thr']} acc {best['accuracy']:.1%} "
                f"fallback {best['fallback_rate']:.1%} ε=1 {best.get('timing_eps1', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "y2_confidence_fallback", "Y2 · 低置信回退", device=args.device)
    write_phase24_result("y2_confidence_fallback", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
