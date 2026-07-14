#!/usr/bin/env python3
"""V2 · 写回 schedule 推理（4 步后停写回）+ M2 head，不重训 Coconut。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase20_common import (
    CAP, FINE_GRID, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase20_result,
)
from coconut_feedback import apply_feedback_config, default_feedback_strategies
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import calibrate_rich_threshold, evaluate_rich_stop, split_train_val_samples

STRATEGIES = [s for s in default_feedback_strategies() if s["id"] in ("baseline", "zero_after4", "residual_zero4")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        _, test_set = load_splits()
        full_set = load_full_dataset()
        train_set, _ = load_splits()
        _, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))

        sweep = []
        best_row = None
        best_key = None
        for strat in STRATEGIES:
            apply_feedback_config(model, strat)
            for min_n in (2, 3):
                thr, cal = calibrate_rich_threshold(
                    head, model, tokenizer, val_sub, cap=CAP, min_n=min_n, device=device, seed=SEED,
                    predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                    eval_profile=profile, optimize="timing", min_accuracy=0.863, thresholds=FINE_GRID,
                )
                row = evaluate_rich_stop(
                    head, model, tokenizer, val_sub, cap=CAP, min_n=min_n, threshold=thr,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                row["params"]["feedback_id"] = strat["id"]
                row["params"]["uses_oracle"] = False
                pt = {
                    "feedback_id": strat["id"], "min_n": min_n, "threshold": thr,
                    "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"),
                    "mean_stop_n": row.get("mean_stop_n"), "feasible": is_feasible(row),
                }
                sweep.append(pt)
                key = ((pt["stop_timing_acc"] or 0), pt["accuracy"])
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = {**pt, "calibration": cal, "feedback_cfg": strat}

        assert best_row
        apply_feedback_config(model, best_row["feedback_cfg"])
        full_row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=best_row["min_n"],
            threshold=best_row["threshold"], device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=best_row["min_n"],
            threshold=best_row["threshold"], device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_row["params"]["feedback_id"] = best_row["feedback_id"]
        full_row["params"]["uses_oracle"] = False
        test_row["params"]["feedback_id"] = best_row["feedback_id"]
        test_row["params"]["uses_oracle"] = False

        return {
            "sweep": sweep,
            "best": {k: best_row[k] for k in ("feedback_id", "min_n", "threshold", "accuracy", "stop_timing_acc")},
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "insight": "Exp9：4 步后停写回减噪声；配合 min_n=2 抬 timing ceiling。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "v2_writeback_infer", "V2 · writeback 推理", device=args.device)
    import json
    write_phase20_result("v2_writeback_infer", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
