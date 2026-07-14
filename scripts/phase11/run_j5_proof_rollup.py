#!/usr/bin/env python3
"""J5 · 汇总五层证明链（引用 M3 + J1–J4）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase11_common import proof_status, timed_run, write_phase11_result

M3 = ROOT / "results" / "phase10" / "m3_extra_steps_ablation_latest.json"
PHASE10_OUT = ROOT / "outbox/results/from_a800/phase10/m3_extra_steps_ablation_latest.json"


def _load(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        m3 = _load(M3) or _load(PHASE10_OUT) or {}
        j1 = _load(ROOT / "results" / "phase11" / "j1_feasible_calibrate_latest.json") or {}
        j2 = _load(ROOT / "results" / "phase11" / "j2_joint_warmstart_latest.json") or {}
        j3 = _load(ROOT / "results" / "phase11" / "j3_joint_deep_latest.json") or {}
        j4 = _load(ROOT / "results" / "phase11" / "j4_generalization_latest.json") or {}

        ab = m3.get("ablation") or {}
        necessity = (
            (ab.get("pct_improved") or 1) < 0.05
            and (ab.get("pct_degraded") or 0) > 0.05
        )
        learnable = bool(j1.get("proof", {}).get("learnable") or (j2.get("test") or {}).get("accuracy", 0) >= 0.85)
        deploy = bool(j2.get("feasible") or j3.get("feasible") or j4.get("feasible"))
        useful = False
        if j4.get("full_419"):
            r = j4["full_419"]["test"]
            useful = (r.get("mean_stop_n") or 8) < 7.5 and r.get("accuracy", 0) >= (j4["full_419"].get("auto_route_acc") or 0) - 0.01
        general = bool((j4.get("proof") or {}).get("general"))

        proof = proof_status(
            necessity=necessity,
            learnable=learnable,
            deploy_feasible=deploy,
            useful=useful,
            general=general,
        )
        layers = [
            {"layer": "1_必要性", "source": "M3", "pass": necessity,
             "detail": f"救回{ab.get('pct_improved', 0):.1%} · 搞砸{ab.get('pct_degraded', 0):.1%}"},
            {"layer": "2_可学习", "source": "M2/J1", "pass": learnable,
             "detail": f"J1 feasible={j1.get('feasible')}"},
            {"layer": "3_可部署可行", "source": "J2/J3", "pass": deploy,
             "detail": f"J2={j2.get('feasible')} J3={j3.get('feasible')}"},
            {"layer": "4_有用", "source": "J4 full419", "pass": useful,
             "detail": f"feasible={j4.get('full_419', {}).get('feasible')}"},
            {"layer": "5_泛化", "source": "J4 alt split", "pass": general,
             "detail": f"seed77 feasible={j4.get('alt_split_seed77', {}).get('feasible')}"},
        ]
        return {
            "proof": proof,
            "layers": layers,
            "fully_proven": proof["fully_proven"],
            "insight": "五层全过 → 方案完整证明；否则看哪层卡住。",
            "device": args.device,
        }

    path = timed_run(run_body, "j5_proof_rollup", "J5 · 证明汇总", device=args.device)
    import json as _json
    write_phase11_result("j5_proof_rollup", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
