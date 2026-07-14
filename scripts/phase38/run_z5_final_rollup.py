#!/usr/bin/env python3
"""Z5 · Phase 38 终局 rollup。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_json(phase: int, name: str) -> dict | None:
    for base in (ROOT / "results" / f"phase{phase}", ROOT / f"outbox/results/from_a800/phase{phase}"):
        path = base / f"{name}_latest.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    from _phase38_common import write_phase38_result

    z1 = load_json(38, "z1_hybrid_seed_robust")
    z2 = load_json(38, "z2_seed_flip_forensic")
    z3 = load_json(38, "z3_hurts_six_audit")
    z4 = load_json(38, "z4_deploy_spec_v4_validate")
    p37 = load_json(37, "y6_final_rollup")

    hy_ok = (z1 or {}).get("hybrid_dual_ok_count", 0)
    tz_ok = (z1 or {}).get("tri_zone_dual_ok_count", 0)
    upgrade_hybrid = hy_ok >= tz_ok

    payload = {
        "experiment_id": "z5_final_rollup",
        "title": "Z5 · Phase 38 终局汇总",
        "phase37_status": (p37 or {}).get("project_lock"),
        "z1_seed_robust": {
            "hybrid_dual_ok_count": hy_ok,
            "tri_zone_dual_ok_count": tz_ok,
            "hybrid_beats_tri_zone_seeds": (z1 or {}).get("hybrid_beats_tri_zone_seeds"),
            "by_seed": (z1 or {}).get("by_seed"),
        },
        "z2_flip_count": (z2 or {}).get("flip_count"),
        "z3_hurt_count": (z3 or {}).get("hurt_count"),
        "z4_deploy_spec_v4": (z4 or {}).get("deploy_spec_v4"),
        "deploy_recommendation": {
            "prosqa": "confidence_fallback τ=0.48",
            "cross_dataset": "hybrid_slice_router" if upgrade_hybrid else "tri_zone",
            "canonical_seed": 99,
            "seed_caveat": "跨集 dual_ok 目前仅在 seed=99 稳定；部署需固定 eval seed 或多 seed 聚合",
        },
        "project_status": "cross_locked_v4" if (z4 or {}).get("ok") else "pending",
        "missing": [x for x, v in [("z1", z1), ("z2", z2), ("z3", z3), ("z4", z4)] if v is None],
        "ok": len([x for x, v in [("z1", z1), ("z2", z2), ("z3", z3), ("z4", z4)] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase38_result("z5_final_rollup", payload)

    md = ROOT / "results" / "phase38" / "PHASE38_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 38 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if z1:
        lines.append(
            f"## Z1 seed 稳健性\n"
            f"- hybrid dual_ok: {hy_ok}/4 seeds\n"
            f"- tri_zone dual_ok: {tz_ok}/4 seeds\n\n"
        )
    if z4:
        cross = ((z4.get("deploy_spec_v4") or {}).get("cross_dataset_ood") or {})
        lines.append(
            f"## deploy_spec_v4\n"
            f"- policy: hybrid_slice_router\n"
            f"- dual_ok: {cross.get('dual_ok')}\n"
            f"- in-dist: {cross.get('in_dist_weighted_delta_pp')} pp\n"
            f"- OOD: {cross.get('ood_weighted_delta_pp')} pp\n"
            f"- hurts: {cross.get('hurts_count')}\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
