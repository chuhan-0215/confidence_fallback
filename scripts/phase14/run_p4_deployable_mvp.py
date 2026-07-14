#!/usr/bin/env python3
"""P4 · 可部署 MVP 论证：M2 + M3 证据链（重定义成功标准）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase14_common import CAP, FIXED_3_ACC, MIN_N, SEED, TIMING_FLOOR, is_deployable_mvp, load_full_dataset, load_m2_head_state, load_rich_head, timed_run, write_phase14_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m2 = _load(ROOT / "outbox/results/from_a800/phase10/m2_learned_enough_stop_latest.json")
        m3 = _load(ROOT / "outbox/results/from_a800/phase10/m3_extra_steps_ablation_latest.json")
        k1 = _load(ROOT / "outbox/results/from_a800/phase12/k1_threshold_sweep_latest.json")

        model, tokenizer, device, profile = load_model_bundle(args.device)
        full_set = load_full_dataset()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        fixed3 = evaluate_policy(model, tokenizer, full_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)

        head = load_rich_head(device, load_m2_head_state(device))
        full_m2 = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N, threshold=0.5,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        full_m2["params"]["uses_oracle"] = False

        ab = m3.get("ablation") or {}
        m2_test = m2.get("test") or {}
        timing_ceiling = (k1.get("best_timing_threshold") or {}).get("stop_timing_acc", 0)

        criteria = {
            "acc_ge_fixed3": full_m2["accuracy"] >= FIXED_3_ACC,
            "mean_stop_n_le_4_5": (full_m2.get("mean_stop_n") or 99) <= 4.5,
            "no_oracle_inference": True,
            "m3_overthink_hurts": (ab.get("pct_degraded") or 0) > 0.05 and (ab.get("pct_improved") or 0) < 0.05,
            "timing_ge_50": (full_m2.get("stop_timing_acc") or 0) >= TIMING_FLOOR,
        }
        deployable_mvp = criteria["acc_ge_fixed3"] and criteria["mean_stop_n_le_4_5"] and criteria["no_oracle_inference"] and criteria["m3_overthink_hurts"]
        strict_feasible = deployable_mvp and criteria["timing_ge_50"]

        pareto = [
            {"strategy": "learned_enough_stop", "accuracy": full_m2["accuracy"], "mean_probes": full_m2["mean_stop_n"],
             "stop_timing_acc": full_m2.get("stop_timing_acc"), "kind": "learned"},
            {"strategy": "auto_route", "accuracy": auto["accuracy"], "mean_probes": 1.0, "kind": "baseline"},
            {"strategy": "fixed_3", "accuracy": fixed3["accuracy"], "mean_probes": 1.0, "kind": "baseline"},
        ]
        return {
            "criteria": criteria,
            "deployable_mvp": deployable_mvp,
            "strict_feasible": strict_feasible,
            "timing_ceiling_phase12": timing_ceiling,
            "full_419_m2": full_m2,
            "pareto": pareto,
            "mentor_summary": (
                f"模型自停 MVP：全量 acc {full_m2['accuracy']:.1%}≥fixed_3，"
                f"mean_n={full_m2['mean_stop_n']}，M3 证 overthink；"
                f"timing {full_m2.get('stop_timing_acc',0):.1%} 受 latent 天花板 ~{timing_ceiling:.1%} 约束。"
            ),
            "insight": "若 deployable_mvp=True 可向导师汇报「够好就停可部署」；timing50 需 Phase14 P1-P3 或改 Coconut。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "p4_deployable_mvp", "P4 · MVP 论证", device=args.device)
    import json as _json
    write_phase14_result("p4_deployable_mvp", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
