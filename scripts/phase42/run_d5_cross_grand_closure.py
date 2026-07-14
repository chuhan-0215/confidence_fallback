#!/usr/bin/env python3
"""D5 · 跨集 grand closure rollup。"""
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
    from _phase42_common import write_phase42_result

    p41 = load_json(41, "c5_absolute_closure_rollup")
    d1 = load_json(42, "d1_seed43_in_dist_forensic")
    d2 = load_json(42, "d2_hybrid_v5_seed_robust")
    d3 = load_json(42, "d3_deploy_bounds_v5")
    d4 = load_json(42, "d4_deploy_spec_v8_validate")

    v4_ok = (d2 or {}).get("v4_dual_ok_count") or (p41 or {}).get("c2_v4_dual_ok_count", 3)
    v5_ok = (d2 or {}).get("v5_dual_ok_count", 0)
    variant = (d4 or {}).get("selected_variant") or "v4"

    payload = {
        "experiment_id": "d5_cross_grand_closure",
        "title": "D5 · 跨集 grand closure",
        "phase41_v4_dual_ok": v4_ok,
        "phase41_status": (p41 or {}).get("project_status"),
        "d1_in_dist_neg": (d1 or {}).get("in_dist_neg_count"),
        "d2_v4_dual_ok": v4_ok,
        "d2_v5_dual_ok": v5_ok,
        "d3_bounds": (d3 or {}).get("deploy_bounds"),
        "d4_deploy_spec_v8": (d4 or {}).get("deploy_spec_v8"),
        "deploy_recommendation": {
            "prosqa": "confidence_fallback τ=0.48 @ seed=99",
            "cross_dataset": (d4 or {}).get("deploy_spec_v8", {}).get("cross_dataset_ood", {}).get("policy", "hybrid_slice_router_v4"),
            "per_seed_dual_ok": f"{max(v4_ok, v5_ok)}/4",
            "canonical_seed": 99,
        },
        "project_status": "cross_grand_final" if (d4 or {}).get("ok") else "cross_absolute_final",
        "missing": [x for x, v in [("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4)] if v is None],
        "ok": len([x for x, v in [("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4)] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase42_result("d5_cross_grand_closure", payload)

    md = ROOT / "results" / "phase42" / "PHASE42_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 42 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if d2:
        lines.append(f"## D2 v5 vs v4\n- v4: {v4_ok}/4\n- v5: {v5_ok}/4\n\n")
    if d4:
        cross = ((d4.get("deploy_spec_v8") or {}).get("cross_dataset_ood") or {})
        lines.append(f"## deploy_spec_v8 ({variant})\n- dual_ok: {cross.get('dual_ok')}\n")
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
