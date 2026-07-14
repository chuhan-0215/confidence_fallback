#!/usr/bin/env python3
"""A4 · deploy spec v2：纳入 P24 confidence_fallback 冠军。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase25_common import SEED, load_json, timed_run, write_phase25_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        y4 = load_json("phase24/y4_deploy_spec_latest.json")
        y2 = load_json("phase24/y2_confidence_fallback_latest.json")
        y3 = load_json("phase24/y3_error_taxonomy_latest.json")
        a1 = load_json("phase25/a1_fallback_finetune_latest.json")

        base = dict((y4 or {}).get("deploy_spec") or {})
        best_y2 = (y2 or {}).get("best") or {}
        a1_best = (a1 or {}).get("best_thr_row") or best_y2

        spec = {
            **base,
            "version": "deploy_v2_20260705",
            "acc_champion": {
                "id": "confidence_fallback",
                "label": "structure_d 主路径 + 低置信回退 knn+M2",
                "accuracy": a1_best.get("accuracy", best_y2.get("accuracy", 0.9499)),
                "fallback_thr": (a1_best.get("params") or best_y2.get("params") or {}).get("fallback_thr", 0.5),
                "fallback_rate": a1_best.get("fallback_rate", best_y2.get("fallback_rate", 0.0883)),
                "timing_eps1": a1_best.get("timing_eps1", best_y2.get("timing_eps1", 0.5074)),
                "uses_oracle": False,
                "note": "P24 Y2 突破；A1 精调后更新",
            },
            "primary": base.get("primary", {}),
            "error_ceiling": {
                "union_acc": y3.get("baseline_union_acc", 0.9594),
                "both_wrong": y3.get("both_wrong", 17),
                "struct_only": y3.get("struct_only_correct", 14),
                "knn_only": y3.get("knn_only_correct", 5),
            },
            "recommended_seed": SEED,
        }
        if a1.get("acc_stats"):
            spec["acc_champion"]["accuracy_mean"] = a1["acc_stats"].get("mean")
            spec["acc_champion"]["accuracy_stdev"] = a1["acc_stats"].get("stdev")

        return {
            "deploy_spec": spec,
            "sources": ["phase17", "phase22", "phase23", "phase24", "phase25"],
            "mentor_brief": (
                f"spec v2：acc 冠军 confidence_fallback {spec['acc_champion']['accuracy']:.1%}；"
                f"简单主推 structure_d μ={spec['primary'].get('accuracy_mean', 0):.1%}。"
            ),
            "insight": "终稿 deploy spec，双轨：简单 structure_d / 高 acc fallback。",
            "device": args.device,
        }

    path = timed_run(run_body, "a4_deploy_spec_v2", "A4 · spec v2", device=args.device)
    write_phase25_result("a4_deploy_spec_v2", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
