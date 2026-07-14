#!/usr/bin/env python3
"""W4 · 汇总 + 选最优 timing/acc 配置。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase18_common import FIXED_3_ACC, TIMING_FLOOR, load_json, timed_run, write_phase18_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        w1 = load_json("phase18/w1_joint_fc_latest.json")
        w2 = load_json("phase18/w2_joint_fc_writeback_latest.json")
        w3 = load_json("phase18/w3_fc_long_head_latest.json")
        baseline = load_json("phase16/r1_mvp_final_latest.json")

        candidates = []
        for tag, data in (("w1_joint_fc", w1), ("w2_writeback", w2), ("w3_fc_long", w3)):
            full = data.get("full_419") or {}
            if full:
                candidates.append({
                    "id": tag,
                    "accuracy": full.get("accuracy"),
                    "timing": full.get("stop_timing_acc"),
                    "mean_n": full.get("mean_stop_n"),
                    "feasible": data.get("feasible"),
                    "deployable_mvp": data.get("deployable_mvp"),
                })

        best_timing = max(candidates, key=lambda c: ((c.get("timing") or 0), c.get("accuracy") or 0), default={})
        best_acc = max(candidates, key=lambda c: c.get("accuracy") or 0, default={})
        feasible_any = any(c.get("feasible") for c in candidates)

        bl_full = ((baseline.get("variants") or {}).get("best_deployable_mvp") or {}).get("full_419") or {}
        delta_timing = (best_timing.get("timing") or 0) - (bl_full.get("stop_timing_acc") or 0)

        return {
            "candidates": candidates,
            "best_timing": best_timing,
            "best_acc": best_acc,
            "feasible_any": feasible_any,
            "baseline_phase16": {
                "accuracy": bl_full.get("accuracy"),
                "timing": bl_full.get("stop_timing_acc"),
            },
            "delta_timing_vs_phase16": delta_timing,
            "mentor_brief": (
                f"Phase18 冲 timing：最优 {best_timing.get('id')} timing {best_timing.get('timing',0):.1%} "
                f"acc {best_timing.get('accuracy',0):.1%}；"
                f"较 Phase16 {'+' if delta_timing>=0 else ''}{delta_timing*100:.1f}pp；"
                f"feasible(timing≥50%)={feasible_any}。"
            ),
            "insight": "若 W1/W2 timing>40% 则写回+joint 有效；否则需更深 Coconut 改造。",
            "device": args.device,
        }

    path = timed_run(run_body, "w4_proof_rollup", "W4 · 汇总", device=args.device)
    import json
    write_phase18_result("w4_proof_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
