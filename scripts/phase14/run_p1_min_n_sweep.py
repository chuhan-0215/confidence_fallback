#!/usr/bin/env python3
"""P1 · M2 冻结头 · min_n×threshold 全扫描（找 timing 甜点）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase14_common import CAP, FINE_GRID, SEED, is_deployable_mvp, is_feasible, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase14_result
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
        _, test_set = load_splits()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)

        state = load_m2_head_state(device)
        if not state:
            raise FileNotFoundError("需要 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        sweep = []
        for min_n in (2, 3, 4):
            for thr in FINE_GRID:
                row = evaluate_rich_stop(
                    head, model, tokenizer, test_set, cap=CAP, min_n=min_n, threshold=thr,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                row["params"]["uses_oracle"] = False
                sweep.append({
                    "min_n": min_n, "threshold": thr,
                    "accuracy": row["accuracy"],
                    "stop_timing_acc": row.get("stop_timing_acc"),
                    "mean_stop_n": row.get("mean_stop_n"),
                    "feasible": is_feasible(row),
                    "deployable_mvp": is_deployable_mvp(row),
                })

        best_timing = max(sweep, key=lambda p: ((p["stop_timing_acc"] or 0), p["accuracy"]))
        best_mvp = max(sweep, key=lambda p: (p["deployable_mvp"], p["accuracy"], p["stop_timing_acc"] or 0))
        best_acc = max(sweep, key=lambda p: p["accuracy"])
        feasible_pts = [p for p in sweep if p["feasible"]]
        return {
            "sweep": sweep,
            "feasible_points": feasible_pts,
            "feasible": bool(feasible_pts),
            "best_timing": best_timing,
            "best_acc": best_acc,
            "best_deployable_mvp": best_mvp,
            "deployable_mvp_count": sum(1 for p in sweep if p["deployable_mvp"]),
            "auto_route_test_acc": auto["accuracy"],
            "insight": "Phase13 证伪改标签/改推理；P1 扫 min_n 看 timing 能否破 35% 天花板。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "p1_min_n_sweep", "P1 · min_n 扫描", device=args.device)
    import json
    write_phase14_result("p1_min_n_sweep", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
