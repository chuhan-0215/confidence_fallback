#!/usr/bin/env python3
"""R3 · 全量 Pareto 对比：min3 / kNN / hybrid / 基线。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase16_common import (
    CAP, MIN_N_BEST, SEED, is_deployable_mvp, is_feasible,
    load_full_dataset, load_json_candidates, load_k3_head_state,
    load_m2_head_state, load_rich_head, pick_q1_configs, timed_run, write_phase16_result,
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
        q1 = pick_q1_configs()
        r2 = load_json_candidates("phase15/q2_knn_min3_combo_latest.json", "phase16/r2_knn_min3_full419_latest.json")
        bt = q1.get("best_timing") or {}
        ba = q1.get("best_deployable_mvp") or {}

        model, tokenizer, device, profile = load_model_bundle(args.device)
        full_set = load_full_dataset()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        fixed3 = evaluate_policy(model, tokenizer, full_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)

        m2_head = load_rich_head(device, load_m2_head_state(device))
        k3_state = load_k3_head_state(device)
        k3_head = load_rich_head(device, k3_state) if k3_state else None

        def _run(head, min_n, thr, label):
            row = evaluate_rich_stop(
                head, model, tokenizer, full_set, cap=CAP, min_n=min_n, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["params"]["uses_oracle"] = False
            return {
                "strategy": label,
                "config": {"min_n": min_n, "threshold": thr},
                "accuracy": row["accuracy"],
                "stop_timing_acc": row.get("stop_timing_acc"),
                "mean_stop_n": row.get("mean_stop_n"),
                "feasible": is_feasible(row),
                "deployable_mvp": is_deployable_mvp(row),
            }

        pareto = [
            _run(m2_head, MIN_N_BEST, float(bt.get("threshold") or 0.35), "min3_best_timing"),
            _run(m2_head, MIN_N_BEST, float(ba.get("threshold") or 0.5), "min3_best_acc"),
            _run(m2_head, 2, 0.5, "m2_baseline"),
        ]
        if k3_head:
            pareto.append(_run(k3_head, MIN_N_BEST, 0.15, "hybrid_min3"))
        if r2.get("best_deployable_mvp"):
            bm = r2["best_deployable_mvp"]
            pareto.append({
                "strategy": "knn_min3_full",
                "config": {"threshold": bm.get("threshold")},
                "accuracy": bm.get("accuracy"),
                "stop_timing_acc": bm.get("stop_timing_acc"),
                "mean_stop_n": bm.get("mean_stop_n"),
                "feasible": bm.get("feasible"),
                "deployable_mvp": bm.get("deployable_mvp"),
            })
        elif r2.get("test"):
            t = r2["test"]
            pareto.append({
                "strategy": "knn_min3_test_only",
                "config": {"threshold": r2.get("threshold")},
                "accuracy": t.get("accuracy"),
                "stop_timing_acc": t.get("stop_timing_acc"),
                "mean_stop_n": t.get("mean_stop_n"),
                "feasible": r2.get("feasible"),
                "deployable_mvp": r2.get("deployable_mvp"),
            })
        pareto.extend([
            {"strategy": "auto_route", "accuracy": auto["accuracy"], "mean_stop_n": 1.0, "kind": "baseline"},
            {"strategy": "fixed_3", "accuracy": fixed3["accuracy"], "mean_stop_n": 1.0, "kind": "baseline"},
        ])

        learned = [p for p in pareto if p.get("kind") != "baseline"]
        best_mvp = max(learned, key=lambda p: (p.get("deployable_mvp", False), p.get("accuracy", 0)))
        best_timing = max(learned, key=lambda p: (p.get("stop_timing_acc") or 0, p.get("accuracy", 0)))

        return {
            "pareto": pareto,
            "best_deployable_mvp": best_mvp,
            "best_timing": best_timing,
            "any_deployable_mvp": any(p.get("deployable_mvp") for p in learned),
            "insight": "全量 419 Pareto 前沿；选导师汇报最优策略。",
            "eval_split": "full_419",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "r3_pareto_full419", "R3 · Pareto", device=args.device)
    import json
    write_phase16_result("r3_pareto_full419", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
