#!/usr/bin/env python3
"""T1 · M2 头 + head∨stable 推理（不重训，OR 非 AND）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase13_common import (
    CAP, FINE_GRID, MIN_N, PATIENCE_GRID, SEED, is_feasible, load_m2_head_state,
    load_rich_head, load_splits, timed_run, write_phase13_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop, split_train_val_samples
from stop_head_tracks import evaluate_rich_or_stable_stop


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
        fixed3 = evaluate_policy(model, tokenizer, test_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)

        state = load_m2_head_state(device)
        if state is None:
            raise FileNotFoundError("需要 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        baseline = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=0.5,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )

        grid = []
        for thr in FINE_GRID:
            for patience in PATIENCE_GRID:
                row = evaluate_rich_or_stable_stop(
                    head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N,
                    threshold=thr, patience=patience, device=device, seed=SEED,
                    predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                grid.append({
                    "threshold": thr, "patience": patience,
                    "val_accuracy": row["accuracy"],
                    "val_stop_timing_acc": row.get("stop_timing_acc"),
                    "val_feasible": is_feasible(row),
                })

        val_feasible = [g for g in grid if g["val_feasible"]]
        if val_feasible:
            pick = max(val_feasible, key=lambda g: (g["val_stop_timing_acc"], g["val_accuracy"]))
        else:
            pick = max(grid, key=lambda g: ((g["val_stop_timing_acc"] or 0), g["val_accuracy"]))

        test_row = evaluate_rich_or_stable_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N,
            threshold=pick["threshold"], patience=pick["patience"],
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row["strategy"] = "head_or_stable"
        return {
            "baseline_m2_correctness": baseline,
            "calibration_grid_size": len(grid),
            "val_feasible_count": len(val_feasible),
            "picked": pick,
            "test": test_row,
            "feasible": is_feasible(test_row),
            "fixed_3_test_acc": fixed3["accuracy"],
            "insight": "推理改 head∨stable（OR）；Phase11 streak 是 AND 导致 timing=0。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "t1_head_or_stable", "T1 · head∨stable", device=args.device)
    import json
    write_phase13_result("t1_head_or_stable", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
