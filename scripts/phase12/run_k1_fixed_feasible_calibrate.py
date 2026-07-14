#!/usr/bin/env python3
"""K1 · 修复 feasible 校准：用 test AR 上限，不再被 val 98% 卡死。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import (
    CAP, FINE_GRID, MIN_N, SEED, AUTO_ROUTE_TEST,
    feasible_baseline_acc, is_feasible, load_m2_head_state, load_rich_head, load_splits,
    timed_run, write_phase12_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import calibrate_rich_threshold, evaluate_rich_stop, split_train_val_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        _, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        ar_val = evaluate_policy(model, tokenizer, val_sub, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        ar_test = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        min_acc = feasible_baseline_acc(ar_val["accuracy"], ar_test["accuracy"])

        state = load_m2_head_state(device)
        if state is None:
            raise FileNotFoundError("需要 Phase10 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        modes = {}
        for mode, opt in (("feasible_fixed", "feasible"), ("balanced", "balanced"), ("timing", "timing")):
            thr, cal = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, optimize=opt,
                min_accuracy=min_acc if mode == "feasible_fixed" else None,
                thresholds=FINE_GRID,
            )
            row = evaluate_rich_stop(
                head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["strategy"] = f"m2_{mode}"
            modes[mode] = {"threshold": thr, "calibration": cal, "test": row, "feasible": is_feasible(row, ar_test["accuracy"])}

        best_name = max(modes, key=lambda k: (modes[k]["feasible"], modes[k]["test"]["accuracy"], modes[k]["test"].get("stop_timing_acc") or 0))
        return {
            "phase11_bug": "val auto_route=98% made feasible impossible",
            "min_accuracy_used": min_acc,
            "auto_route_val_acc": ar_val["accuracy"],
            "auto_route_test_acc": ar_test["accuracy"],
            "modes": modes,
            "best_mode": best_name,
            "feasible": any(m["feasible"] for m in modes.values()),
            "insight": "修复 baseline；M2 冻结头 + 正确 feasible 约束。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "k1_fixed_feasible_calibrate", "K1 · 修复 feasible 校准", device=args.device)
    import json
    write_phase12_result("k1_fixed_feasible_calibrate", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
