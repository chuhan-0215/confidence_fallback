#!/usr/bin/env python3
"""K5 · Phase 12 证明汇总（修正标准）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import timed_run, write_phase12_result

PHASE10_M3 = ROOT / "outbox/results/from_a800/phase10/m3_extra_steps_ablation_latest.json"
PHASE10_M2 = ROOT / "outbox/results/from_a800/phase10/m2_learned_enough_stop_latest.json"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = _load(PHASE10_M3) or _load(ROOT / "results/phase10/m3_extra_steps_ablation_latest.json") or {}
        m2 = _load(PHASE10_M2) or _load(ROOT / "results/phase10/m2_learned_enough_stop_latest.json") or {}
        k1 = _load(ROOT / "results" / "phase12/k1_threshold_sweep_latest.json") or {}
        k2 = _load(ROOT / "results" / "phase12/k2_stable_correct_train_latest.json") or {}
        k3h = _load(ROOT / "results/phase12/k3_hybrid_distill_latest.json") or {}
        k3l = _load(ROOT / "results/phase12/k3_long_head_train_latest.json") or {}
        k4 = _load(ROOT / "results/phase12/k4_generalization_latest.json") or {}

        ab = m3.get("ablation") or {}
        necessity = (ab.get("pct_improved") or 1) < 0.05 and (ab.get("pct_degraded") or 0) > 0.05
        m2_acc = (m2.get("test") or {}).get("accuracy", 0)
        learnable = m2_acc >= 0.85
        deploy = bool(k1.get("feasible") or k2.get("feasible") or k3h.get("feasible") or k3l.get("feasible"))
        useful = bool(k4.get("feasible"))

        layers = [
            {"layer": "1_必要性", "pass": necessity, "detail": f"M3 救回{ab.get('pct_improved',0):.1%} 搞砸{ab.get('pct_degraded',0):.1%}"},
            {"layer": "2_可学习", "pass": learnable, "detail": f"M2 acc={m2_acc:.1%}"},
            {"layer": "3_可行", "pass": deploy, "detail": f"K1={k1.get('feasible')} K2={k2.get('feasible')} K3h={k3h.get('feasible')} K3l={k3l.get('feasible')}"},
            {"layer": "4_泛化", "pass": useful, "detail": f"K4 feasible={k4.get('feasible')}"},
        ]
        fully = all(l["pass"] for l in layers)
        return {
            "layers": layers,
            "fully_proven": fully,
            "feasible_criterion": "acc>=fixed_3(86.3%) AND timing>=50%",
            "phase11_bug": "Phase11 误用 auto_route(98% val) 作 feasible 门槛，导致校准失效",
            "insight": "M2 87.5% 已过 fixed_3；核心瓶颈是 timing 35%→50%。",
            "device": args.device,
        }

    path = timed_run(run_body, "k5_proof_rollup", "K5 · 证明汇总", device=args.device)
    import json as _json
    write_phase12_result("k5_proof_rollup", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
