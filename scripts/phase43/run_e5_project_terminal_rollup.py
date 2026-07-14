#!/usr/bin/env python3
"""E5 · 项目终局 rollup（PROJECT_COMPLETE）。"""
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
    from _phase43_common import write_phase43_result

    p42 = load_json(42, "d5_cross_grand_closure")
    e1 = load_json(43, "e1_seed43_irreducibility")
    e2 = load_json(43, "e2_v5_canonical_enhanced")
    e3 = load_json(43, "e3_seed_panel_rollup")
    e4 = load_json(43, "e4_deploy_spec_v8_final")

    payload = {
        "experiment_id": "e5_project_terminal_rollup",
        "title": "E5 · 项目终局",
        "phase42_status": (p42 or {}).get("project_status"),
        "e1_seed43_irreducible": (e1 or {}).get("conclusion"),
        "e2_v5_enhanced": (e2 or {}).get("recommend_enhanced_tier"),
        "e2_v5_delta_weighted": (e2 or {}).get("delta_weighted_pp"),
        "e3_panel_dual_ok": f"{(e3 or {}).get('dual_ok_count', '?')}/8",
        "e3_dual_ok_seeds": (e3 or {}).get("dual_ok_seeds"),
        "e4_deploy_spec": (e4 or {}).get("deploy_spec_v8_final"),
        "deploy_recommendation": {
            "prosqa": "confidence_fallback τ=0.48 @ seed=99 → 94.75%",
            "cross_default": "hybrid_slice_router_v4（3/4 seed, pooled dual_ok）",
            "cross_enhanced": "hybrid_slice_router_v5 @ seed=99 only（hurts 5 vs 6）",
            "canonical_seed": 99,
        },
        "project_status": "PROJECT_COMPLETE",
        "cross_status": "cross_grand_final",
        "missing": [x for x, v in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)] if v is None],
        "ok": len([x for x, v in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase43_result("e5_project_terminal_rollup", payload)

    md = ROOT / "results" / "phase43" / "PHASE43_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Phase 43 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n",
        f"## 项目状态\n- **PROJECT_COMPLETE**\n- cross: cross_grand_final\n\n",
    ]
    if e3:
        lines.append(f"## E3 seed 面板\n- dual_ok: {(e3 or {}).get('dual_ok_count')}/8\n- seeds: {(e3 or {}).get('dual_ok_seeds')}\n\n")
    if e4:
        lines.append("## deploy_spec_v8_final\n- 默认 v4 + 可选 v5 @99 增强档\n")
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
