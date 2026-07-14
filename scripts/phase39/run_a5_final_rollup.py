#!/usr/bin/env python3
"""A5 · Phase 39 终局 rollup。"""
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
    from _phase39_common import write_phase39_result

    a1 = load_json(39, "a1_fixed_profile_cross")
    a2 = load_json(39, "a2_hybrid_v2_seed_robust")
    a3 = load_json(39, "a3_multiseed_pooled")
    a4 = load_json(39, "a4_deploy_spec_v5_validate")
    p38 = load_json(38, "z5_final_rollup")

    v2_ok = (a2 or {}).get("v2_dual_ok_count", 0)
    v4_ok = (a2 or {}).get("v4_dual_ok_count", 0)
    upgrade_v5 = v2_ok >= v4_ok

    payload = {
        "experiment_id": "a5_final_rollup",
        "title": "A5 · Phase 39 终局汇总",
        "phase38_deploy_v4": (p38 or {}).get("z4_deploy_spec_v4"),
        "a1_fixed_profile": (a1 or {}).get("results"),
        "a1_fixed_improves": (a1 or {}).get("fixed_improves_robustness"),
        "a2_v4_dual_ok_count": v4_ok,
        "a2_v2_dual_ok_count": v2_ok,
        "a3_pooled_dual_ok": (a3 or {}).get("pooled_dual_ok"),
        "a4_deploy_spec_v5": (a4 or {}).get("deploy_spec_v5"),
        "deploy_recommendation": {
            "prosqa": "confidence_fallback τ=0.48",
            "cross_dataset": "hybrid_slice_router_v2" if upgrade_v5 else "hybrid_slice_router (v4)",
            "eval_profile_cross": "default (coconut + random) + seed=99",
            "canonical_seed": 99,
            "fixed_edges": "REJECTED (A1 0/4 dual_ok)",
        },
        "project_status": "cross_locked_v5" if (a4 or {}).get("ok") else "pending",
        "missing": [x for x, v in [("a1", a1), ("a2", a2), ("a3", a3), ("a4", a4)] if v is None],
        "ok": len([x for x, v in [("a1", a1), ("a2", a2), ("a3", a3), ("a4", a4)] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase39_result("a5_final_rollup", payload)

    md = ROOT / "results" / "phase39" / "PHASE39_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 39 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if a1:
        lines.append(
            f"## A1 fixed profile\n"
            f"- default dual_ok seeds: {(a1.get('results') or {}).get('default', {}).get('dual_ok_count')}/4\n"
            f"- fixed_edges dual_ok seeds: {(a1.get('results') or {}).get('fixed_edges', {}).get('dual_ok_count')}/4\n\n"
        )
    if a2:
        lines.append(f"## A2 hybrid v2\n- v4 dual_ok: {v4_ok}/4\n- v2 dual_ok: {v2_ok}/4\n\n")
    if a3:
        lines.append(f"## A3 pooled\n- {json.dumps(a3.get('pooled_dual_ok'), ensure_ascii=False)}\n\n")
    if a4:
        cross = ((a4.get("deploy_spec_v5") or {}).get("cross_dataset_ood") or {})
        lines.append(
            f"## deploy_spec_v5\n"
            f"- dual_ok: {cross.get('dual_ok')}\n"
            f"- hurts: {cross.get('hurts_count')}\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
