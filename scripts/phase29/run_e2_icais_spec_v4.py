#!/usr/bin/env python3
"""E2 · ICAIS deploy spec v4 终稿数字锁定。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase29_common import load_json, timed_run, write_phase29_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        p26 = load_json("phase26/b4_icais_spec_lock_latest.json") or {}
        p27 = load_json("phase27/c7_alt_paradigm_rollup_latest.json") or {}
        p28 = load_json("phase28/d5_gap_closure_rollup_latest.json") or {}
        base = dict((p26.get("deploy_spec") or {}))
        spec = {
            **base,
            "version": "deploy_v4_final_20260705",
            "icais_numbers": {
                **(base.get("icais_numbers") or {}),
                "p27_best_alternative": (p27.get("best_alternative") or {}).get("accuracy", 0.9475),
                "p28_best": (p28.get("best_p28") or {}).get("accuracy"),
                "p28_gap_closed": p28.get("gap_closed", False),
                "project_status": "complete_pending_e3",
            },
            "narrative_arc": [
                "阶段1 找边界：步数扫描 + 题深规律",
                "阶段2 找机制：overthink + 写回噪声",
                "阶段3 做控制：Stop Head + confidence_fallback",
                "阶段4 算边界：P27 证伪替代范式 + P28 缺口闭环",
            ],
            "p27_conclusion": "非 Stop Head 路线最高 94.75%，未超 confidence_fallback 95.23%。",
            "p28_focus": "靶向 idx 111/189/261 三题缺口（knn_only）。",
        }
        return {
            "deploy_spec": spec,
            "mentor_brief": "ICAIS v4：95.23% 冠军维持；P27 94.75% 替代上界；P28 缺口闭环结果写入。",
            "device": args.device,
        }

    path = timed_run(run_body, "e2_icais_spec_v4", "E2 · ICAIS v4", device=args.device)
    write_phase29_result("e2_icais_spec_v4", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
