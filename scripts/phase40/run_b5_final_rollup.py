#!/usr/bin/env python3
"""B5 · Phase 40 终局 rollup。"""
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
    from _phase40_common import write_phase40_result

    b1 = load_json(40, "b1_worst_seed_slice_audit")
    b2 = load_json(40, "b2_hybrid_v3_seed_robust")
    b3 = load_json(40, "b3_deploy_bounds")
    b4 = load_json(40, "b4_deploy_spec_v6_validate")
    p39 = load_json(39, "a5_final_rollup")

    v3_ok = (b2 or {}).get("v3_dual_ok_count", 0)
    v5_ok = (b2 or {}).get("v5_dual_ok_count", 0)
    upgrade_v6 = v3_ok > v5_ok or ((b4 or {}).get("selected_variant") == "v3")

    payload = {
        "experiment_id": "b5_final_rollup",
        "title": "B5 · Phase 40 终局汇总",
        "phase39_status": (p39 or {}).get("project_status"),
        "phase39_fixed_edges_rejected": (p39 or {}).get("a1_fixed_improves") is False,
        "b1_flip_count": (b1 or {}).get("flip_count"),
        "b1_in_dist_hurts": (b1 or {}).get("in_dist_hurt_count"),
        "b2_v5_dual_ok_count": v5_ok,
        "b2_v3_dual_ok_count": v3_ok,
        "b3_deploy_bounds": (b3 or {}).get("deploy_bounds"),
        "b4_deploy_spec_v6": (b4 or {}).get("deploy_spec_v6"),
        "deploy_recommendation": {
            "prosqa": "confidence_fallback τ=0.48",
            "cross_dataset": (b4 or {}).get("deploy_spec_v6", {}).get("cross_dataset_ood", {}).get("policy", "hybrid_slice_router_v2"),
            "eval_profile_cross": "default (coconut + random) + seed=99",
            "canonical_seed": 99,
            "fixed_edges": "REJECTED (P39 A1)",
        },
        "project_status": "cross_locked_v6" if (b4 or {}).get("ok") else ("cross_locked_v5" if not upgrade_v6 else "pending"),
        "missing": [x for x, v in [("b1", b1), ("b2", b2), ("b3", b3), ("b4", b4)] if v is None],
        "ok": len([x for x, v in [("b1", b1), ("b2", b2), ("b3", b3), ("b4", b4)] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase40_result("b5_final_rollup", payload)

    md = ROOT / "results" / "phase40" / "PHASE40_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 40 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if b1:
        lines.append(f"## B1 worst-seed audit\n- flips: {b1.get('flip_count')}\n- in-dist hurts @43: {b1.get('in_dist_hurt_count')}\n\n")
    if b2:
        lines.append(f"## B2 v3 vs v5\n- v5 dual_ok: {v5_ok}/4\n- v3 dual_ok: {v3_ok}/4\n\n")
    if b3:
        lines.append(f"## B3 deploy bounds\n```json\n{json.dumps(b3.get('deploy_bounds'), ensure_ascii=False, indent=2)}\n```\n\n")
    if b4:
        cross = ((b4.get("deploy_spec_v6") or {}).get("cross_dataset_ood") or {})
        lines.append(
            f"## deploy_spec_v6 ({b4.get('selected_variant')})\n"
            f"- dual_ok: {cross.get('dual_ok')}\n"
            f"- hurts: {cross.get('hurts_count')}\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
