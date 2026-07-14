#!/usr/bin/env python3
"""Q5 · Phase 15 汇总。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase15_common import load_json_candidates, timed_run, write_phase15_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = load_json_candidates("phase10/m3_extra_steps_ablation_latest.json")
        m2 = load_json_candidates("phase10/m2_learned_enough_stop_latest.json")
        p1 = load_json_candidates("phase14/p1_min_n_sweep_latest.json")
        q1 = load_json_candidates("phase15/q1_min3_full419_latest.json")
        q2 = load_json_candidates("phase15/q2_knn_min3_combo_latest.json")
        q3 = load_json_candidates("phase15/q3_hybrid_min3_distill_latest.json")
        q4 = load_json_candidates("phase15/q4_mvp_v2_latest.json")

        ab = m3.get("ablation") or {}
        p1_bt = p1.get("best_timing") or {}
        layers = [
            {"layer": "1_必要性", "pass": (ab.get("pct_improved") or 1) < 0.05 and (ab.get("pct_degraded") or 0) > 0.05,
             "detail": f"M3 救回{ab.get('pct_improved',0):.1%} 搞砸{ab.get('pct_degraded',0):.1%}"},
            {"layer": "2_可学习", "pass": (m2.get("test") or {}).get("accuracy", 0) >= 0.85,
             "detail": f"M2 acc={(m2.get('test') or {}).get('accuracy',0):.1%}"},
            {"layer": "3_P14突破", "pass": (p1_bt.get("stop_timing_acc") or 0) >= 0.38,
             "detail": f"P1 min_n=3 timing {p1_bt.get('stop_timing_acc',0):.1%} acc {p1_bt.get('accuracy',0):.1%}"},
            {"layer": "4_timing50", "pass": bool(q1.get("feasible") or q2.get("feasible") or q3.get("feasible")),
             "detail": f"Q1={q1.get('feasible')} Q2={q2.get('feasible')} Q3={q3.get('feasible')}"},
            {"layer": "5_deployable_mvp", "pass": bool(q4.get("deployable_mvp")),
             "detail": q4.get("mentor_summary", "")},
        ]
        return {
            "layers": layers,
            "fully_proven_strict": all(l["pass"] for l in layers[:4]),
            "deployable_mvp": q4.get("deployable_mvp"),
            "phase14_breakthrough": f"min_n=3 timing {p1_bt.get('stop_timing_acc',0):.1%}",
            "phase15_strategy": "min3全量 + kNN组合 + hybrid+min3 + MVP v2",
            "insight": q4.get("mentor_summary", "见 Q4。"),
            "device": args.device,
        }

    path = timed_run(run_body, "q5_proof_rollup", "Q5 · 汇总", device=args.device)
    import json as _json
    write_phase15_result("q5_proof_rollup", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
