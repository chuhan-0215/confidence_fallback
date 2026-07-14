#!/usr/bin/env python3
"""K1 · M2 冻结头阈值全扫描（可行标准 fixed_3=86.3% + timing≥50%）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import CAP, FINE_GRID, FIXED_3_ACC, MIN_N, SEED, is_feasible, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase12_result
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
        fixed3 = evaluate_policy(model, tokenizer, test_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)
        auto = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)

        state = load_m2_head_state(device)
        if state is None:
            raise FileNotFoundError("需要 Phase10 m2_enough_stop_head.pt")
        head = load_rich_head(device, state)

        sweep = []
        for thr in FINE_GRID:
            row = evaluate_rich_stop(
                head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            sweep.append({
                "threshold": thr,
                "accuracy": row["accuracy"],
                "stop_timing_acc": row.get("stop_timing_acc"),
                "mean_stop_n": row.get("mean_stop_n"),
                "feasible_fixed3": is_feasible(row),
            })

        feasible_pts = [p for p in sweep if p["feasible_fixed3"]]
        best_acc = max(sweep, key=lambda p: p["accuracy"])
        best_timing = max(sweep, key=lambda p: (p.get("stop_timing_acc") or 0, p["accuracy"]))
        return {
            "sweep": sweep,
            "feasible_points": feasible_pts,
            "feasible": bool(feasible_pts),
            "best_acc_threshold": best_acc,
            "best_timing_threshold": best_timing,
            "fixed_3_test_acc": fixed3["accuracy"],
            "auto_route_test_acc": auto["accuracy"],
            "feasible_criterion": f"acc>={FIXED_3_ACC} timing>=0.5",
            "insight": "Phase11 误用 auto_route(92.9%) 作门槛；官方标准是 fixed_3(86.3%)。M2 87.5% 已过 acc 线，差 timing。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "k1_threshold_sweep", "K1 · 阈值扫描", device=args.device)
    import json
    write_phase12_result("k1_threshold_sweep", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
