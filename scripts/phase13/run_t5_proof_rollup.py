#!/usr/bin/env python3
"""T5 · Phase 13 证明汇总。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase13_common import timed_run, write_phase13_result

M3 = ROOT / "outbox/results/from_a800/phase10/m3_extra_steps_ablation_latest.json"
M2 = ROOT / "outbox/results/from_a800/phase10/m2_learned_enough_stop_latest.json"
K1 = ROOT / "outbox/results/from_a800/phase12/k1_threshold_sweep_latest.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = _load(M3) or _load(ROOT / "results/phase10/m3_extra_steps_ablation_latest.json")
        m2 = _load(M2) or _load(ROOT / "results/phase10/m2_learned_enough_stop_latest.json")
        k1 = _load(K1) or _load(ROOT / "results/phase12/k1_threshold_sweep_latest.json")
        t1 = _load(ROOT / "results/phase13/t1_head_or_stable_latest.json")
        t2 = _load(ROOT / "results/phase13/t2_first_correct_train_latest.json")
        t3 = _load(ROOT / "results/phase13/t3_earliest_stop_train_latest.json")
        t4 = _load(ROOT / "results/phase13/t4_generalization_latest.json")

        ab = m3.get("ablation") or {}
        necessity = (ab.get("pct_improved") or 1) < 0.05 and (ab.get("pct_degraded") or 0) > 0.05
        learnable = (m2.get("test") or {}).get("accuracy", 0) >= 0.85
        k1_ceiling = k1.get("best_timing_threshold") or {}
        timing_ceiling = k1_ceiling.get("stop_timing_acc", 0)
        deploy = bool(t1.get("feasible") or t2.get("feasible") or t3.get("feasible"))
        general = bool(t4.get("feasible"))

        layers = [
            {"layer": "1_必要性", "pass": necessity, "detail": f"M3 救回{ab.get('pct_improved',0):.1%} 搞砸{ab.get('pct_degraded',0):.1%}"},
            {"layer": "2_可学习", "pass": learnable, "detail": f"M2 acc={(m2.get('test') or {}).get('accuracy',0):.1%}"},
            {"layer": "3_K12天花板", "pass": False, "detail": f"K1 最高 timing {timing_ceiling:.1%}（14阈值无可行点）"},
            {"layer": "4_P13可行", "pass": deploy, "detail": f"T1={t1.get('feasible')} T2={t2.get('feasible')} T3={t3.get('feasible')}"},
            {"layer": "5_泛化", "pass": general, "detail": f"T4 feasible={t4.get('feasible')}"},
        ]
        return {
            "layers": layers,
            "fully_proven": all(l["pass"] for l in layers),
            "phase12_conclusion": "is_correct+调参 timing 天花板 ~35%",
            "phase13_strategy": "改推理(head∨stable) + 改标签(first_correct/earliest_stop)",
            "insight": "若 T1/T2 仍不过 50% timing → 需端到端 timing loss 或改 Coconut。",
            "device": args.device,
        }

    path = timed_run(run_body, "t5_proof_rollup", "T5 · 汇总", device=args.device)
    import json as _json
    write_phase13_result("t5_proof_rollup", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
