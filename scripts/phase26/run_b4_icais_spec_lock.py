#!/usr/bin/env python3
"""B4 · ICAIS / deploy spec v3 数字锁定。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase26_common import load_json, timed_run, write_phase26_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        p25 = load_json("phase25/a4_deploy_spec_v2_latest.json")
        a1 = load_json("phase25/a1_fallback_finetune_latest.json")
        p24 = load_json("phase24/y3_error_taxonomy_latest.json")
        base = dict((p25 or {}).get("deploy_spec") or {})
        spec = {
            **base,
            "version": "deploy_v3_final_20260705",
            "icais_numbers": {
                "baseline_fixed3": 0.863,
                "auto_route": 0.931,
                "acc_champion": 0.9523,
                "acc_champion_thr": 0.48,
                "acc_champion_mean": (a1.get("acc_stats") or {}).get("mean", 0.9389),
                "structure_d_mean": 0.9375,
                "union_ceiling": 0.9594,
                "timing_strict_max": 0.37,
                "timing_eps1_champion": 0.5074,
            },
            "narrative_arc": [
                "阶段1 找边界：步数扫描 + 题深规律",
                "阶段2 找机制：overthink + 写回噪声",
                "阶段3 做控制：Stop Head + confidence_fallback",
                "阶段4 算边界：timing 37% 硬上限 + 17 题双错",
            ],
        }
        return {
            "deploy_spec": spec,
            "mentor_brief": "ICAIS 定稿数字：86.3→93.1→95.2%；timing 37%；并集 95.94%。",
            "insight": "投稿/海报直接引用 icais_numbers。",
            "device": args.device,
        }

    path = timed_run(run_body, "b4_icais_spec_lock", "B4 · ICAIS 定稿", device=args.device)
    write_phase26_result("b4_icais_spec_lock", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
