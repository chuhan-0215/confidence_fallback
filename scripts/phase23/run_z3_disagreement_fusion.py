#!/usr/bin/env python3
"""Z3 · 分歧融合：structure_d 与 knn 答案不一致时用不对称规则仲裁。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase23_common import (
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_json, load_m2_head_state, load_rich_head, load_splits,
    timed_run, timing_metrics, write_phase23_result,
)
from boundary_budget import (
    _asymmetry_bin, blind_depth, build_prompt_budget_labels,
    make_d4_knn_budget_fn, make_structure_budget_fn, train_d4_knn_bank,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, extract_latent_hidden,
    first_correct_step, split_train_val_samples,
)


@torch.no_grad()
def predict_knn_answer(head, model, tokenizer, sample, device, seed, profile, thr, floor_fn):
    floor_n = max(MIN_N, min(CAP, floor_fn(sample)))
    fc, preds = first_correct_step(
        model, tokenizer, sample, cap=CAP, device=device, seed=seed,
        predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
        expected_fn=expected_answer, eval_profile=profile,
    )
    stop_n = CAP
    final = preds.get(CAP, "")
    prev, streak = "", 0
    for n in range(1, CAP + 1):
        final = preds[n]
        ab, streak, ch = _rich_step_features(final, prev, streak)
        prev = final
        prompt = build_eval_prompt(sample, n, seed=seed + n,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
        prob = torch.sigmoid(head(
            hid.unsqueeze(0), torch.tensor([n], device=device),
            torch.tensor([ab], device=device), torch.tensor([streak], device=device),
            torch.tensor([ch], device=device),
        )).item()
        stop_n = n
        if n >= floor_n and prob >= thr:
            break
    return final, stop_n, fc


@torch.no_grad()
def eval_fusion(model, tokenizer, head, samples, device, seed, profile, thr, knn_floor, tie_mode: str):
    struct_fn = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    correct = agree = disagree = struct_pick = knn_pick = tie_pick = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n_s = struct_fn(sample)
        pred_s = predict_at_n(model, tokenizer, sample, n_s, device, seed=seed + idx * 31, eval_profile=profile)
        pred_k, stop_k, fc = predict_knn_answer(head, model, tokenizer, sample, device, seed + idx * 31, profile, thr, knn_floor)
        if pred_s == pred_k:
            final, stop_n = pred_s, n_s
            agree += 1
        else:
            disagree += 1
            d = blind_depth(sample)
            if tie_mode == "asymmetry":
                # 4跳+候选深度不同 → 信 knn（更保守 d-1 倾向）；否则信 structure
                if d >= 4 and _asymmetry_bin(sample, cap=CAP):
                    final, stop_n = pred_k, stop_k
                    knn_pick += 1
                else:
                    final, stop_n = pred_s, n_s
                    struct_pick += 1
            elif tie_mode == "trust_knn":
                final, stop_n = pred_k, stop_k
                knn_pick += 1
            else:
                final, stop_n = pred_s, n_s
                struct_pick += 1
        if final == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "agree": agree, "disagree": disagree,
        "struct_pick_on_disagree": struct_pick, "knn_pick_on_disagree": knn_pick,
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {"uses_oracle": False, "tie_mode": tie_mode},
    }
    row.update(timing_metrics(stop_ns, fcs))
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
        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        knn_bank, _ = train_d4_knn_bank(train_rows, feature_key="joint_features")
        knn_floor = make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP)
        thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, thresholds=FINE_GRID,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile, optimize="accuracy",
        )

        results = []
        for mode in ("trust_struct", "trust_knn", "asymmetry"):
            row = eval_fusion(model, tokenizer, head, full, device, SEED, profile, thr, knn_floor, mode)
            row["deployable_mvp"] = is_deployable_mvp(row)
            row["eps_deployable"] = is_eps_deployable(row)
            results.append(row)

        best = max(results, key=lambda r: r["accuracy"])
        p22 = load_json("phase22/x5_complement_audit_latest.json")
        return {
            "results": results,
            "best": best,
            "full_419": best,
            "baseline_union_acc": p22.get("union_acc"),
            "baseline_structure_d": p22.get("structure_d_acc"),
            "insight": "P22 并集 95.9% 上界；分歧融合能否无 oracle 逼近？",
            "mentor_brief": (
                f"Z3 分歧融合：最优 {best['params']['tie_mode']} acc {best['accuracy']:.1%}；"
                f"并集上界 {p22.get('union_acc', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "z3_disagreement_fusion", "Z3 · 分歧融合", device=args.device)
    write_phase23_result("z3_disagreement_fusion", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
