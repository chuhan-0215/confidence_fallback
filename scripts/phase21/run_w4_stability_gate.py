#!/usr/bin/env python3
"""W4 · 稳定性门控：跳数 floor + 答案连续稳定 k 步即停，不用 head。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase21_common import (
    CAP, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_splits, timed_run, write_phase21_result,
)
from boundary_budget import blind_depth, build_prompt_budget_labels, make_d4_rich_binary_budget_fn, train_d4_weighted_binary_mlp
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, first_correct_step, split_train_val_samples

MIN_N = 2
STREAK_GRID = (2, 3)


@torch.no_grad()
def evaluate_stability_gate(
    model, tokenizer, samples, *, cap, streak_min, device, seed,
    predict_fn, expected_fn, eval_profile, floor_fn, global_min_n,
):
    correct = total = stop_sum = timing_hits = timing_total = 0
    stop_hist = {}
    for idx, sample in enumerate(samples):
        expected = expected_fn(sample, eval_profile)
        floor_n = max(global_min_n, min(cap, floor_fn(sample)))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        stop_n = cap
        final_pred = preds.get(cap, "")
        prev, streak = "", 0
        for n in range(1, cap + 1):
            final_pred = preds[n]
            _, streak, _ = _rich_step_features(final_pred, prev, streak)
            prev = final_pred
            stop_n = n
            if n >= floor_n and streak >= streak_min:
                break
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
    return {
        "accuracy": round(acc, 4), "correct": correct, "total": total,
        "mean_stop_n": round(stop_sum / total, 2) if total else 0.0,
        "stop_n_histogram": stop_hist,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "stop_timing_hits": timing_hits, "stop_timing_total": timing_total,
        "params": {
            "streak_min": streak_min, "cap": cap, "global_min_n": global_min_n,
            "mode": "stability_gate", "uses_oracle": False,
        },
        "strategy": "stability_gate",
    }


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

        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        val_rows = build_prompt_budget_labels(
            model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        rich_head, rich_meta = train_d4_weighted_binary_mlp(train_rows, val_rows, feature_key="rich_features")

        floor_variants = [
            ("blind_depth", lambda s: blind_depth(s)),
            ("rich_mlp_floor", make_d4_rich_binary_budget_fn(rich_head, min_n=MIN_N, cap=CAP, device=device)),
        ]

        sweep = []
        for floor_name, floor_fn in floor_variants:
            for streak_min in STREAK_GRID:
                for split_name, samples in (("test", test_set), ("full_419", full)):
                    row = evaluate_stability_gate(
                        model, tokenizer, samples, cap=CAP, streak_min=streak_min,
                        device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                        eval_profile=profile, floor_fn=floor_fn, global_min_n=MIN_N,
                    )
                    sweep.append({
                        "floor": floor_name, "streak_min": streak_min, "split": split_name,
                        "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"),
                        "mean_stop_n": row.get("mean_stop_n"),
                        "feasible": is_feasible(row),
                        "deployable_mvp": is_deployable_mvp(row),
                    })

        full_rows = [r for r in sweep if r["split"] == "full_419"]
        best = max(full_rows, key=lambda r: ((r.get("stop_timing_acc") or 0), r["accuracy"]))
        best_mvp = max(full_rows, key=lambda r: (r["deployable_mvp"], r["accuracy"]))
        return {
            "sweep": sweep,
            "rich_meta": rich_meta,
            "best": best,
            "best_deployable_mvp": best_mvp,
            "full_419": best,
            "feasible": any(r["feasible"] for r in full_rows),
            "deployable_mvp": any(r["deployable_mvp"] for r in full_rows),
            "insight": "稳定性门控：不依赖 M2 head，测「答案自稳定」作为通用停步信号是否可行。",
            "mentor_brief": (
                f"W4 稳定性门控：最优 {best['floor']} streak={best['streak_min']} "
                f"timing {best.get('stop_timing_acc', 0):.1%} acc {best['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "w4_stability_gate", "W4 · 稳定性门控", device=args.device)
    write_phase21_result("w4_stability_gate", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
