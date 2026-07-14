#!/usr/bin/env python3
"""K2 · stable_correct 标签训练（答对且稳定 2 步 → 停，对齐 timing）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import CAP, FINE_GRID, MIN_N, SEED, is_feasible, load_splits, timed_run, write_phase12_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import (
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    evaluate_rich_stop,
    split_train_val_samples,
    train_rich_stop_head,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        fixed3 = evaluate_policy(model, tokenizer, test_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="stable_correct",
        )
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="stable_correct",
        )
        head, train_metrics = train_rich_stop_head(train_ex, val_ex, epochs=args.epochs, device=device)
        ckpt = ROOT / "results" / "phase12" / "k2_stable_correct_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics, "label_mode": "stable_correct"}, ckpt)

        results = {}
        for mode, opt in (("balanced", "balanced"), ("timing", "timing"), ("feasible", "feasible")):
            min_acc = 0.863 if mode == "feasible" else None
            thr, cal = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, optimize=opt, min_accuracy=min_acc, thresholds=FINE_GRID,
            )
            row = evaluate_rich_stop(
                head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["strategy"] = f"stable_correct_{mode}"
            results[mode] = {"threshold": thr, "calibration": cal, "test": row, "feasible": is_feasible(row)}

        best = max(results.values(), key=lambda r: (r["feasible"], r["test"]["accuracy"], r["test"].get("stop_timing_acc") or 0))
        return {
            "train_metrics": train_metrics,
            "label_mode": "stable_correct",
            "variants": results,
            "test": best["test"],
            "feasible": any(r["feasible"] for r in results.values()),
            "fixed_3_test_acc": fixed3["accuracy"],
            "insight": "标签：答对且连续 2 步稳定才停，更贴近 first_correct timing。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "k2_stable_correct_train", "K2 · stable_correct", device=args.device)
    import json
    write_phase12_result("k2_stable_correct_train", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
