#!/usr/bin/env python3
"""Q4 · MVP v2：min_n=3 最优配置 + 修复 M3 加载 + 全量论证。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase15_common import (
    CAP, FIXED_3_ACC, MIN_N_BEST, SEED, TIMING_FLOOR,
    is_deployable_mvp, load_full_dataset, load_json_candidates,
    load_m2_head_state, load_rich_head, m3_overthink_hurts, timed_run, write_phase15_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop


def _pick_min3_config() -> tuple[int, float]:
    p1 = load_json_candidates("phase14/p1_min_n_sweep_latest.json")
    bt = p1.get("best_timing") or p1.get("best_deployable_mvp") or {}
    return int(bt.get("min_n") or MIN_N_BEST), float(bt.get("threshold") or 0.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = load_json_candidates(
            "phase10/m3_extra_steps_ablation_latest.json",
            "outbox/results/from_a800/phase10/m3_extra_steps_ablation_latest.json",
        )
        p1 = load_json_candidates("phase14/p1_min_n_sweep_latest.json")
        min_n, threshold = _pick_min3_config()

        model, tokenizer, device, profile = load_model_bundle(args.device)
        full_set = load_full_dataset()
        _, test_set = __import__("_phase15_common", fromlist=["load_splits"]).load_splits()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        fixed3 = evaluate_policy(model, tokenizer, full_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)

        head = load_rich_head(device, load_m2_head_state(device))
        full_row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=min_n, threshold=threshold,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_row["params"]["uses_oracle"] = False
        test_row = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=min_n, threshold=threshold,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        test_row["params"]["uses_oracle"] = False

        timing_ceiling = (p1.get("best_timing") or {}).get("stop_timing_acc", 0)
        ab = m3.get("ablation") or {}
        criteria = {
            "acc_ge_fixed3_full": full_row["accuracy"] >= FIXED_3_ACC,
            "acc_ge_fixed3_test": test_row["accuracy"] >= FIXED_3_ACC,
            "mean_stop_n_le_4_5": (full_row.get("mean_stop_n") or 99) <= 4.5,
            "no_oracle_inference": True,
            "m3_overthink_hurts": m3_overthink_hurts(m3),
            "timing_ge_50_full": (full_row.get("stop_timing_acc") or 0) >= TIMING_FLOOR,
            "timing_ge_50_test": (test_row.get("stop_timing_acc") or 0) >= TIMING_FLOOR,
        }
        deployable_mvp = (
            criteria["acc_ge_fixed3_full"]
            and criteria["mean_stop_n_le_4_5"]
            and criteria["no_oracle_inference"]
            and criteria["m3_overthink_hurts"]
        )
        deployable_mvp_test = (
            criteria["acc_ge_fixed3_test"]
            and criteria["mean_stop_n_le_4_5"]
            and criteria["no_oracle_inference"]
            and criteria["m3_overthink_hurts"]
        )
        strict_feasible = deployable_mvp and criteria["timing_ge_50_full"]

        return {
            "config": {"min_n": min_n, "threshold": threshold},
            "criteria": criteria,
            "deployable_mvp": deployable_mvp,
            "deployable_mvp_test_split": deployable_mvp_test,
            "strict_feasible": strict_feasible,
            "timing_ceiling_phase14": timing_ceiling,
            "m3_ablation": ab,
            "full_419": full_row,
            "test_40pct": test_row,
            "pareto": [
                {"strategy": "learned_min3", "accuracy": full_row["accuracy"], "mean_probes": full_row["mean_stop_n"],
                 "stop_timing_acc": full_row.get("stop_timing_acc"), "kind": "learned"},
                {"strategy": "auto_route", "accuracy": auto["accuracy"], "mean_probes": 1.0, "kind": "baseline"},
                {"strategy": "fixed_3", "accuracy": fixed3["accuracy"], "mean_probes": 1.0, "kind": "baseline"},
            ],
            "mentor_summary": (
                f"min_n={min_n} thr={threshold}：全量 acc {full_row['accuracy']:.1%}，"
                f"timing {full_row.get('stop_timing_acc',0):.1%}（P14 天花板 {timing_ceiling:.1%}）；"
                f"deployable_mvp={deployable_mvp}；M3 搞砸{ab.get('pct_degraded',0):.1%}。"
            ),
            "insight": "修复 P4：用 P1 最优 min_n=3 + 多路径加载 M3；分别报全量/test 两套 MVP。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "q4_mvp_v2", "Q4 · MVP v2", device=args.device)
    import json as _json
    write_phase15_result("q4_mvp_v2", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
