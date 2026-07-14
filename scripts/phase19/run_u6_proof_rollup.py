#!/usr/bin/env python3
"""U6 · Phase 19 汇总。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase19_common import load_json, timed_run, write_phase19_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        exps = ["u1_patience_stop", "u2_streak_and_stop", "u3_learned_hybrid", "u4_head_and_conv"]
        candidates = []
        for eid in exps:
            d = load_json(f"phase19/{eid}_latest.json")
            full = d.get("full_419") or {}
            if full:
                candidates.append({
                    "id": eid,
                    "accuracy": full.get("accuracy"),
                    "timing": full.get("stop_timing_acc"),
                    "mean_n": full.get("mean_stop_n"),
                    "feasible": d.get("feasible"),
                })
        p16 = load_json("phase16/r1_mvp_final_latest.json")
        bl = ((p16.get("variants") or {}).get("best_timing") or {}).get("full_419") or {}
        best = max(candidates, key=lambda c: ((c.get("timing") or 0), c.get("accuracy") or 0), default={})
        u5 = load_json("phase19/u5_failure_analysis_latest.json")

        return {
            "candidates": candidates,
            "best": best,
            "baseline_phase16_timing": bl.get("stop_timing_acc"),
            "failure_reports": u5.get("reports"),
            "feasible_any": any(c.get("feasible") for c in candidates),
            "mentor_brief": (
                f"Phase19 换推理策略：最优 {best.get('id')} timing {best.get('timing',0):.1%} "
                f"acc {best.get('accuracy',0):.1%}；"
                f"基线 Phase16 timing {bl.get('stop_timing_acc',0):.1%}。"
            ),
            "insight": u5.get("insight", ""),
            "device": args.device,
        }

    path = timed_run(run_body, "u6_proof_rollup", "U6 · 汇总", device=args.device)
    import json
    write_phase19_result("u6_proof_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
