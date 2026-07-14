#!/usr/bin/env python3
"""U2 · M2 + streak∧head 停步（稳定且 head 同意，非 OR）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase19_common import (
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase19_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import evaluate_streak_gated_stop, split_train_val_samples


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
        for thr in FINE_GRID:
            for patience in (2, 3):
                row = evaluate_streak_gated_stop(
                    head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, threshold=thr,
                    stable_min_n=MIN_N, patience=patience, device=device, seed=SEED,
                    predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                row["params"]["uses_oracle"] = False
                sweep.append({
                    "threshold": thr, "patience": patience,
                    "accuracy": row["accuracy"], "stop_timing_acc": row.get("stop_timing_acc"),
                    "mean_stop_n": row.get("mean_stop_n"), "feasible": is_feasible(row),
                })

        best = max(sweep, key=lambda p: ((p["stop_timing_acc"] or 0), p["accuracy"]))
        full_row = evaluate_streak_gated_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N, threshold=best["threshold"],
            stable_min_n=MIN_N, patience=best["patience"], device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_row["params"]["uses_oracle"] = False
        test_row = evaluate_streak_gated_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=best["threshold"],
            stable_min_n=MIN_N, patience=best["patience"], device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row["params"]["uses_oracle"] = False
        return {
            "sweep": sweep,
            "best": best,
            "full_419": full_row,
            "test": test_row,
            "feasible": is_feasible(full_row),
            "deployable_mvp": is_deployable_mvp(full_row),
            "insight": "T1 用 OR 崩 acc；U2 用 stable∧head AND 对齐 timing。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "u2_streak_and_stop", "U2 · streak∧head", device=args.device)
    import json
    write_phase19_result("u2_streak_and_stop", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
