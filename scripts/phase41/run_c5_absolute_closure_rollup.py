#!/usr/bin/env python3
"""C5 · Phase 41 终局 rollup（跨集 absolute closure）。"""
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
    from _phase41_common import write_phase41_result

    p40 = load_json(40, "b5_final_rollup")
    c1 = load_json(41, "c1_ood_gap_forensic")
    c2 = load_json(41, "c2_hybrid_v4_seed_robust")
    c3 = load_json(41, "c3_deploy_bounds_v4")
    c4 = load_json(41, "c4_deploy_spec_v7_validate")

    variant = (c4 or {}).get("selected_variant") or "v3"
    v3_strict = ((p40 or {}).get("b3_deploy_bounds") or {}).get("hybrid_v3", {}).get("deploy_ok_strict")

    payload = {
        "experiment_id": "c5_absolute_closure_rollup",
        "title": "C5 · 跨集 absolute closure",
        "phase40_v6_bug": "B4 误锁 v5；B3 已证 v3 pooled dual_ok",
        "phase40_v3_pooled_dual_ok": v3_strict,
        "c1_ood_gap_count": (c1 or {}).get("gap_count"),
        "c2_v3_dual_ok": (c2 or {}).get("v3_dual_ok_count"),
        "c2_v4_dual_ok": (c2 or {}).get("v4_dual_ok_count"),
        "c3_deploy_bounds": (c3 or {}).get("deploy_bounds"),
        "c4_deploy_spec_v7": (c4 or {}).get("deploy_spec_v7"),
        "deploy_recommendation": {
            "prosqa": "confidence_fallback τ=0.48 @ seed=99",
            "cross_dataset": (c4 or {}).get("deploy_spec_v7", {}).get("cross_dataset_ood", {}).get("policy", "hybrid_slice_router_v3"),
            "eval_profile_cross": "default + seed=99",
            "pooled_reference": "v3 pooled dual_ok=true（P40 B3）",
            "per_seed_limit": "dual_ok 仅 seed=99；42/44 OOD<7",
        },
        "project_status": "cross_absolute_final" if (c4 or {}).get("ok") else "cross_locked_v6",
        "missing": [x for x, v in [("c1", c1), ("c2", c2), ("c3", c3), ("c4", c4)] if v is None],
        "ok": len([x for x, v in [("c1", c1), ("c2", c2), ("c3", c3), ("c4", c4)] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase41_result("c5_absolute_closure_rollup", payload)

    md = ROOT / "results" / "phase41" / "PHASE41_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 41 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if c1:
        lines.append(f"## C1 OOD gap\n- gaps: {c1.get('gap_count')}\n\n")
    if c2:
        lines.append(f"## C2 v4 vs v3\n- v3: {(c2 or {}).get('v3_dual_ok_count')}/4\n- v4: {(c2 or {}).get('v4_dual_ok_count')}/4\n\n")
    if c4:
        cross = ((c4.get("deploy_spec_v7") or {}).get("cross_dataset_ood") or {})
        lines.append(f"## deploy_spec_v7 ({variant})\n- dual_ok: {cross.get('dual_ok')}\n- policy: {cross.get('policy')}\n")
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
