#!/usr/bin/env python3
"""X1 · structure_d floor + M2 二段式：Phase21 acc 冠军 + 在线细停 + ε-timing。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase22_common import (
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_eps_deployable, is_feasible,
    load_full_dataset, load_m2_head_state, load_json, load_rich_head, load_splits,
    row_summary, timed_run, timing_metrics, write_phase22_result,
)
from boundary_budget import make_structure_budget_fn
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, extract_latent_hidden,
    first_correct_step, split_train_val_samples,
)


@torch.no_grad()
def evaluate_floor_m2_eps(
    head, model, tokenizer, samples, *, cap, threshold, device, seed,
    predict_fn, expected_fn, build_prompt_fn, eval_profile, floor_fn, global_min_n,
):
    head.eval()
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist = {}
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        floor_n = max(global_min_n, min(cap, floor_fn(sample)))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        fcs.append(fc)
        stop_n = cap
        final_pred = preds.get(cap, "")
        prev, streak = "", 0
        for n in range(1, cap + 1):
            final_pred = preds[n]
            ab, streak, ch = _rich_step_features(final_pred, prev, streak)
            prev = final_pred
            prompt = build_prompt_fn(sample, n, seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0),
                torch.tensor([n], device=device),
                torch.tensor([ab], device=device),
                torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            stop_n = n
            if n >= floor_n and prob >= threshold:
                break
        stop_ns.append(stop_n)
        total += 1
        if final_pred == expected:
            correct += 1
        stop_sum += stop_n
        stop_hist[str(stop_n)] = stop_hist.get(str(stop_n), 0) + 1
        if fc is not None:
            timing_total += 1
            if stop_n == fc:
                timing_hits += 1
    acc = correct / total if total else 0.0
    row = {
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits, "stop_timing_total": timing_total,
        "params": {
            "threshold": threshold, "cap": cap, "global_min_n": global_min_n,
            "mode": "structure_d_floor_m2", "uses_oracle": False,
        },
    }
    row.update(timing_metrics(stop_ns, fcs))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))
        floor_fn = make_structure_budget_fn(min_n=MIN_N, cap=CAP)

        thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N,
            thresholds=FINE_GRID, device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
            optimize="accuracy", min_accuracy=0.90,
        )

        results = []
        for split_name, samples in (("test", test_set), ("full_419", full)):
            row = evaluate_floor_m2_eps(
                head, model, tokenizer, samples, cap=CAP, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
                floor_fn=floor_fn, global_min_n=MIN_N,
            )
            results.append(row_summary(row, "structure_d_floor_m2", split=split_name, threshold=thr))

        full_row = next(r for r in results if r["split"] == "full_419")
        p17 = load_json("phase17/s1_corrected_final_latest.json")
        knn_acc = (p17.get("deployable_mvp") or {}).get("accuracy", 0.926)
        return {
            "results": results,
            "threshold": thr,
            "full_419": full_row,
            "feasible": full_row.get("feasible"),
            "deployable_mvp": full_row.get("deployable_mvp"),
            "eps_deployable": full_row.get("eps_deployable"),
            "baseline_knn_acc": knn_acc,
            "insight": "structure_d 作 floor + M2 细停：能否在保持 93%+ acc 同时抬 ε-timing？",
            "mentor_brief": (
                f"X1 二段式：structure_d+M2 acc {full_row['accuracy']:.1%} "
                f"strict {full_row.get('stop_timing_acc', 0):.1%} ε=1 {full_row.get('timing_eps1', 0):.1%}；"
                f"knn {knn_acc:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "x1_structure_m2", "X1 · structure_d+M2", device=args.device)
    write_phase22_result("x1_structure_m2", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
