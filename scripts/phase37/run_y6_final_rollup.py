#!/usr/bin/env python3
"""Y6 · Phase 37 终局 rollup + 项目锁定。"""
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
    from _phase37_common import write_phase37_result

    y1 = load_json(37, "y1_tri_zone_full_grid")
    y2 = load_json(37, "y2_tri_zone_seed_robust")
    y3 = load_json(37, "y3_hybrid_slice_router")
    y4 = load_json(37, "y4_fallback_wrong_repair")
    y5 = load_json(37, "y5_deploy_spec_v3_validate")
    p36 = load_json(36, "x6_final_rollup")

    best_grid = (y1 or {}).get("best") or {}
    hybrid_better = False
    if y3:
        tz = (y3.get("summaries") or {}).get("tri_zone") or {}
        hy = (y3.get("summaries") or {}).get("hybrid_router") or {}
        hybrid_better = (hy.get("weighted_mean_delta_pp") or 0) > (tz.get("weighted_mean_delta_pp") or 0)

    payload = {
        "experiment_id": "y6_final_rollup",
        "title": "Y6 · Phase 37 终局汇总",
        "phase36_locked_policy": (p36 or {}).get("deploy_spec_v3"),
        "y1_best_grid": best_grid,
        "y1_locked_dual_ok": ((y1 or {}).get("locked_params") or {}).get("dual_ok"),
        "y2_all_seed_dual_ok": (y2 or {}).get("all_dual_ok"),
        "y3_dual_ok": (y3 or {}).get("dual_ok"),
        "y3_hybrid_beats_tri_zone": hybrid_better,
        "y4_gap_pp": (y4 or {}).get("gap_pp"),
        "y5_deploy_spec_v3": (y5 or {}).get("deploy_spec_v3"),
        "project_lock": {
            "cross_dataset_policy": "tri_zone",
            "params": {"t_low": 0.40, "t_mid": 0.48},
            "prosqa_policy": "confidence_fallback",
            "params_prosqa": {"fallback_thr": 0.48},
            "status": "locked" if (y5 or {}).get("ok") else "pending",
        },
        "lessons": [
            "P35 W2 test split 优势未在全量复现（test 幻觉）",
            "combo/agreement 在全量等同，tri_zone 唯一 dual_ok",
            "push_ext6 为能力上限非 collateral",
        ],
        "missing": [x for x, v in [
            ("y1", y1), ("y2", y2), ("y3", y3), ("y4", y4), ("y5", y5),
        ] if v is None],
        "ok": len([x for x, v in [
            ("y1", y1), ("y2", y2), ("y3", y3), ("y4", y4), ("y5", y5),
        ] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase37_result("y6_final_rollup", payload)

    md = ROOT / "results" / "phase37" / "PHASE37_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 37 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if y5:
        ds = y5.get("deploy_spec_v3") or {}
        cross = ds.get("cross_dataset_ood") or {}
        lines.append(
            f"## deploy_spec_v3\n"
            f"- ProsQA acc: {(ds.get('prosqa_in_distribution') or {}).get('accuracy')}\n"
            f"- Cross dual_ok: {cross.get('dual_ok')}\n"
            f"- in-dist: {cross.get('in_dist_weighted_delta_pp')} pp\n"
            f"- OOD: {cross.get('ood_weighted_delta_pp')} pp\n\n"
        )
    if best_grid:
        s = best_grid.get("summary") or {}
        lines.append(
            f"## Y1 best grid t_low={best_grid.get('t_low')}\n"
            f"- dual_ok: {best_grid.get('dual_ok')}\n"
            f"- weighted Δ: {s.get('weighted_mean_delta_pp')} pp\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
