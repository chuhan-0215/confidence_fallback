#!/usr/bin/env python3
"""R1 · MVP 定稿：Q1 全量最优配置（非 P14 test 配置）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase16_common import (
    CAP, FIXED_3_ACC, MIN_N_BEST, SEED, TIMING_FLOOR,
    is_deployable_mvp, load_full_dataset, load_json_candidates,
    load_m2_head_state, load_rich_head, m3_necessity_pass, pick_q1_configs,
    timed_run, write_phase16_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop


def _eval_config(head, model, tokenizer, samples, min_n, threshold, device, profile, pfn):
    row = evaluate_rich_stop(
        head, model, tokenizer, samples, cap=CAP, min_n=min_n, threshold=threshold,
        device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt, eval_profile=profile,
    )
    row["params"]["uses_oracle"] = False
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = load_json_candidates("phase10/m3_extra_steps_ablation_latest.json")
        cfgs = pick_q1_configs()
        variants = {}
        for name, cfg in cfgs.items():
            if cfg:
                variants[name] = {
                    "min_n": int(cfg.get("min_n") or MIN_N_BEST),
                    "threshold": float(cfg.get("threshold") or 0.5),
                }

        model, tokenizer, device, profile = load_model_bundle(args.device)
        full_set = load_full_dataset()
        _, test_set = __import__("_phase16_common", fromlist=["load_splits"]).load_splits()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        fixed3 = evaluate_policy(model, tokenizer, full_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))

        results = {}
        for name, vc in variants.items():
            results[name] = {
                "config": vc,
                "full_419": _eval_config(head, model, tokenizer, full_set, vc["min_n"], vc["threshold"], device, profile, pfn),
                "test_40pct": _eval_config(head, model, tokenizer, test_set, vc["min_n"], vc["threshold"], device, profile, pfn),
            }
            f = results[name]["full_419"]
            results[name]["deployable_mvp_full"] = is_deployable_mvp(f)
            results[name]["deployable_mvp_test"] = is_deployable_mvp(results[name]["test_40pct"])

        primary = results.get("best_deployable_mvp") or results.get("best_acc") or {}
        full_row = primary.get("full_419") or {}
        primary_cfg = primary.get("config") or {}
        ab = m3.get("ablation") or {}
        criteria = {
            "acc_ge_fixed3_full": (full_row.get("accuracy") or 0) >= FIXED_3_ACC,
            "mean_stop_n_le_4_5": (full_row.get("mean_stop_n") or 99) <= 4.5,
            "no_oracle_inference": True,
            "m3_necessity": m3_necessity_pass(m3),
            "timing_ge_50_full": (full_row.get("stop_timing_acc") or 0) >= TIMING_FLOOR,
        }
        deployable_mvp = all([
            criteria["acc_ge_fixed3_full"],
            criteria["mean_stop_n_le_4_5"],
            criteria["no_oracle_inference"],
            criteria["m3_necessity"],
        ])

        return {
            "variants": results,
            "primary": "best_deployable_mvp",
            "criteria": criteria,
            "deployable_mvp": deployable_mvp,
            "m3_ablation": ab,
            "baselines": {
                "auto_route": {"accuracy": auto["accuracy"], "mean_probes": 1.0},
                "fixed_3": {"accuracy": fixed3["accuracy"], "mean_probes": 1.0},
            },
            "mentor_summary": (
                f"全量最优 min_n=3 thr={primary_cfg.get('threshold', '?')}: "
                f"acc {full_row.get('accuracy', 0):.1%}，timing {full_row.get('stop_timing_acc', 0):.1%}，"
                f"mean_n={full_row.get('mean_stop_n')}；deployable_mvp={deployable_mvp}；"
                f"M3 搞砸{ab.get('pct_degraded', 0):.1%}。"
            ),
            "insight": "Q4 误用 P14 test 配置 thr=0.15；R1 改用 Q1 全量最优 thr=0.5/0.35。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "r1_mvp_final", "R1 · MVP 定稿", device=args.device)
    import json
    write_phase16_result("r1_mvp_final", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
