#!/usr/bin/env python3
"""V5 · Phase 20 汇总 + mentor_brief。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase20_common import load_json, timed_run, write_phase20_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        exps = ["v1_adaptive_min_n", "v2_writeback_infer", "v3_fc_min2_train", "v4_min2_baseline"]
        candidates = []
        for eid in exps:
            d = load_json(f"phase20/{eid}_latest.json")
            full = d.get("full_419") or {}
            if full:
                candidates.append({
                    "id": eid,
                    "accuracy": full.get("accuracy"),
                    "timing": full.get("stop_timing_acc"),
                    "mean_n": full.get("mean_stop_n"),
                    "feasible": d.get("feasible"),
                    "deployable_mvp": d.get("deployable_mvp"),
                })
        p19 = load_json("phase19/u6_proof_rollup_latest.json")
        p16 = load_json("phase16/r1_mvp_final_latest.json")
        bl19 = (p19.get("best") or {}).get("timing")
        bl16 = ((p16.get("variants") or {}).get("best_timing") or {}).get("full_419", {}).get("stop_timing_acc")
        best = max(candidates, key=lambda c: ((c.get("timing") or 0), c.get("accuracy") or 0), default={})
        u5 = load_json("phase19/u5_failure_analysis_latest.json")
        ceiling = load_json("phase20/timing_ceiling_analysis.json")

        delta = None
        if best.get("timing") is not None and bl16 is not None:
            delta = round(best["timing"] - bl16, 4)

        brief = (
            f"Phase20 破 min_n 天花板：最优 {best.get('id')} "
            f"timing {best.get('timing', 0):.1%} acc {best.get('accuracy', 0):.1%}；"
            f"Phase19 {bl19:.1%} Phase16 {bl16:.1%}；"
            f"min_n=3 理论上限 {ceiling.get('ceilings', {}).get('3', 0.44):.1%}。"
        )

        return {
            "candidates": candidates,
            "best": best,
            "baseline_phase19_timing": bl19,
            "baseline_phase16_timing": bl16,
            "delta_timing_vs_phase16": delta,
            "failure_reports": u5.get("reports"),
            "timing_ceilings": ceiling.get("ceilings"),
            "feasible_any": any(c.get("feasible") for c in candidates),
            "deployable_mvp_any": any(c.get("deployable_mvp") for c in candidates),
            "mentor_brief": brief,
            "insight": "P19 U5：late_stop 57%（fc=1/2 被 min_n=3 挡）；V1/V3/V4 主攻 min_n=2。",
            "device": args.device,
        }

    path = timed_run(run_body, "v5_proof_rollup", "V5 · 汇总", device=args.device)
    import json
    write_phase20_result("v5_proof_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
