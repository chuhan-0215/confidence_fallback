#!/usr/bin/env python3
"""P5 · Phase 14 汇总。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase14_common import timed_run, write_phase14_result


def _load(rel: str) -> dict:
    for base in (ROOT / "outbox/results/from_a800", ROOT / "results"):
        p = base / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = _load("phase10/m3_extra_steps_ablation_latest.json")
        m2 = _load("phase10/m2_learned_enough_stop_latest.json")
        p1 = _load("phase14/p1_min_n_sweep_latest.json")
        p2 = _load("phase14/p2_fc_oversample_train_latest.json")
        p3 = _load("phase14/p3_prefix_knn_floor_latest.json")
        p4 = _load("phase14/p4_deployable_mvp_latest.json")

        ab = m3.get("ablation") or {}
        layers = [
            {"layer": "必要性", "pass": (ab.get("pct_improved") or 1) < 0.05 and (ab.get("pct_degraded") or 0) > 0.05,
             "detail": f"M3 救回{ab.get('pct_improved',0):.1%} 搞砸{ab.get('pct_degraded',0):.1%}"},
            {"layer": "可学习", "pass": (m2.get("test") or {}).get("accuracy", 0) >= 0.85,
             "detail": f"M2 acc={(m2.get('test') or {}).get('accuracy',0):.1%}"},
            {"layer": "timing50", "pass": bool(p1.get("feasible") or p2.get("feasible") or p3.get("feasible")),
             "detail": f"P1={p1.get('feasible')} P2={p2.get('feasible')} P3={p3.get('feasible')}"},
            {"layer": "deployable_mvp", "pass": bool(p4.get("deployable_mvp")),
             "detail": p4.get("mentor_summary", "")},
        ]
        return {
            "layers": layers,
            "fully_proven_strict": all(l["pass"] for l in layers[:3]),
            "deployable_mvp": p4.get("deployable_mvp"),
            "phase13_conclusion": "改标签/改推理均失败；M2 仍是 Pareto 最优",
            "phase14_strategy": "min_n扫描 + fc过采样 + 前缀floor + MVP重定义",
            "insight": "见 P4 mentor_summary。",
            "device": args.device,
        }

    path = timed_run(run_body, "p5_proof_rollup", "P5 · 汇总", device=args.device)
    import json as _json
    write_phase14_result("p5_proof_rollup", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
