#!/usr/bin/env python3
"""R5 · Phase 16 汇总。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase16_common import load_json_candidates, m3_necessity_pass, timed_run, write_phase16_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = load_json_candidates("phase10/m3_extra_steps_ablation_latest.json")
        m2 = load_json_candidates("phase10/m2_learned_enough_stop_latest.json")
        q1 = load_json_candidates("phase15/q1_min3_full419_latest.json")
        r1 = load_json_candidates("phase16/r1_mvp_final_latest.json")
        r2 = load_json_candidates("phase16/r2_knn_min3_full419_latest.json")
        r4 = load_json_candidates("phase16/r4_mentor_brief_latest.json")

        ab = m3.get("ablation") or {}
        layers = [
            {"layer": "1_必要性", "pass": m3_necessity_pass(m3),
             "detail": f"M3 救回{ab.get('pct_improved',0):.1%} 搞砸{ab.get('pct_degraded',0):.1%}"},
            {"layer": "2_可学习", "pass": (m2.get("test") or {}).get("accuracy", 0) >= 0.85,
             "detail": f"M2 acc={(m2.get('test') or {}).get('accuracy',0):.1%}"},
            {"layer": "3_min_n突破", "pass": (q1.get("best_timing") or {}).get("stop_timing_acc", 0) >= 0.37,
             "detail": f"全量 timing {(q1.get('best_timing') or {}).get('stop_timing_acc',0):.1%}"},
            {"layer": "4_timing50", "pass": bool(r1.get("criteria", {}).get("timing_ge_50_full")),
             "detail": f"stretch goal；天花板 ~39% test"},
            {"layer": "5_deployable_mvp", "pass": bool(r1.get("deployable_mvp")),
             "detail": r4.get("mentor_brief", r1.get("mentor_summary", ""))},
        ]
        return {
            "layers": layers,
            "fully_proven_strict": all(l["pass"] for l in layers[:4]),
            "deployable_mvp": r1.get("deployable_mvp"),
            "knn_full_deployable": (r2.get("deployable_mvp_count") or 0) > 0,
            "phase15_gap": "Q4 误用 thr=0.15；Q1 全量 thr=0.35–0.5 已过 deployable_mvp",
            "phase16_strategy": "MVP定稿 + kNN全量 + Pareto + 导师汇报",
            "insight": r4.get("mentor_brief", ""),
            "device": args.device,
        }

    path = timed_run(run_body, "r5_proof_rollup", "R5 · 汇总", device=args.device)
    import json
    write_phase16_result("r5_proof_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
