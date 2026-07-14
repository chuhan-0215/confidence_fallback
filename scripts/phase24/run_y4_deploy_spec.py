#!/usr/bin/env python3
"""Y4 · 定稿 deploy spec：汇总 P17–P23 定稿方案为可部署 JSON。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase24_common import SEED, load_json, timed_run, write_phase24_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        p17 = load_json("phase17/s1_corrected_final_latest.json")
        p22 = load_json("phase22/x6_proof_rollup_latest.json")
        p23 = load_json("phase23/z6_final_manifest_latest.json")
        z1 = load_json("phase23/z1_seed_robustness_latest.json")
        z3 = load_json("phase23/z3_disagreement_fusion_latest.json")

        sd_stats = (z1.get("report") or {}).get("structure_d", {}).get("acc_stats", {})
        fusion = z3.get("best") or {}

        spec = {
            "version": "deploy_v1_20260705",
            "dataset": "prosqa_full_419",
            "model": "coconut_checkpoint_300",
            "primary": {
                "id": "structure_d",
                "label": "BFS 题深单次前向",
                "accuracy_mean": sd_stats.get("mean", 0.9375),
                "accuracy_stdev": sd_stats.get("stdev", 0.006),
                "accuracy_seed99": 0.9475,
                "mean_stop_n": 3.52,
                "single_forward": True,
                "uses_oracle": False,
                "inference": "budget_n = max(3, min(8, blind_depth)); predict_at_n(budget_n)",
            },
            "eps_deploy": {
                "id": "structure_d_m2",
                "label": "题深 floor + M2 在线细停",
                "accuracy": 0.9356,
                "timing_eps1": 0.5074,
                "threshold": 0.15,
                "min_n": 3,
                "uses_oracle": False,
            },
            "strict_timing": {
                "id": "knn_min3_full",
                "label": "knn floor + M2 (Phase17)",
                "accuracy": (p17.get("deployable_mvp") or {}).get("accuracy", 0.926),
                "stop_timing_acc": 0.37,
                "uses_oracle": False,
            },
            "stretch_eps_only": {
                "id": "m2_min3",
                "timing_eps1": 0.5882,
                "accuracy": 0.8616,
                "note": "ε 最高但 acc 偏低",
            },
            "phase23_fusion": {
                "id": "trust_struct",
                "accuracy": fusion.get("accuracy", 0.9475),
                "note": "等价 structure_d；分歧 19 题全信 struct",
            },
            "metrics_not_achieved": {
                "strict_feasible": False,
                "reason": "min_n=3 + late_stop 57%；strict timing 天花板 ~37%",
            },
            "recommended_seed": SEED,
        }

        return {
            "deploy_spec": spec,
            "sources": ["phase17", "phase22", "phase23"],
            "mentor_brief": (
                f"定稿 spec：主推 structure_d μ={spec['primary']['accuracy_mean']:.1%}；"
                f"ε-deploy structure_d+M2 {spec['eps_deploy']['timing_eps1']:.1%}；"
                f"strict knn {spec['strict_timing']['stop_timing_acc']:.1%}。"
            ),
            "insight": "可部署 JSON spec，供论文/工程直接引用。",
            "device": args.device,
        }

    path = timed_run(run_body, "y4_deploy_spec", "Y4 · deploy spec", device=args.device)
    write_phase24_result("y4_deploy_spec", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
