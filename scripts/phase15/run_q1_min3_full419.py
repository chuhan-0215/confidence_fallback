#!/usr/bin/env python3
"""Q1 · P1 最优 min_n=3 在全量 419 上阈值扫描。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase15_common import (
    CAP, FINE_GRID, MIN_N_BEST, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_m2_head_state, load_rich_head, timed_run, write_phase15_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full_set = load_full_dataset()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        fixed3 = evaluate_policy(model, tokenizer, full_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)

        state = load_m2_head_state(device)
        if not state:
            raise FileNotFoundError("需要 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        sweep = []
        for thr in FINE_GRID:
            row = evaluate_rich_stop(
                head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N_BEST, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["params"]["uses_oracle"] = False
            sweep.append({
                "min_n": MIN_N_BEST, "threshold": thr,
                "accuracy": row["accuracy"],
                "stop_timing_acc": row.get("stop_timing_acc"),
                "mean_stop_n": row.get("mean_stop_n"),
                "feasible": is_feasible(row),
                "deployable_mvp": is_deployable_mvp(row),
            })

        feasible_pts = [p for p in sweep if p["feasible"]]
        best_timing = max(sweep, key=lambda p: ((p["stop_timing_acc"] or 0), p["accuracy"]))
        best_mvp = max(sweep, key=lambda p: (p["deployable_mvp"], p["accuracy"], p["stop_timing_acc"] or 0))
        best_acc = max(sweep, key=lambda p: p["accuracy"])
        return {
            "sweep": sweep,
            "feasible_points": feasible_pts,
            "feasible": bool(feasible_pts),
            "best_timing": best_timing,
            "best_acc": best_acc,
            "best_deployable_mvp": best_mvp,
            "deployable_mvp_count": sum(1 for p in sweep if p["deployable_mvp"]),
            "auto_route_full_acc": auto["accuracy"],
            "fixed_3_full_acc": fixed3["accuracy"],
            "insight": "P1 发现 min_n=3 timing 39%；Q1 验证全量 419 是否 deployable_mvp 且 timing 能否逼近 50%。",
            "eval_split": "full_419",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "q1_min3_full419", "Q1 · min3 全量", device=args.device)
    import json
    write_phase15_result("q1_min3_full419", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
