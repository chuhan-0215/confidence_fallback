#!/usr/bin/env python3
"""V6 · Phase 34 终局 rollup。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase34_common import PHASE34_OUT, write_phase34_result  # noqa: E402


def load_json(name: str) -> dict | None:
    path = PHASE34_OUT / f"{name}_latest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    v1 = load_json("v1_agreement_lock")
    v2 = load_json("v2_tri_zone_sweep")
    v3 = load_json("v3_hop4_tri_zone")
    v4 = load_json("v4_collateral_replay")
    v5 = load_json("v5_champion_validate")
    p33 = json.loads((ROOT / "outbox/results/from_a800/phase33/u6_final_rollup_latest.json").read_text())
    if not (ROOT / "outbox/results/from_a800/phase33/u6_final_rollup_latest.json").is_file():
        p33 = None

    hop4_summary = (v3 or {}).get("summaries", {}).get("hop4_tri_zone")
    payload = {
        "experiment_id": "v6_final_rollup",
        "title": "V6 · Phase 34 终局汇总",
        "phase33_baseline_cross": (p33 or {}).get("u5_cross_summaries", {}).get("baseline"),
        "v1_summaries": (v1 or {}).get("summaries"),
        "v2_best_val": (v2 or {}).get("best_val"),
        "v2_cross_summary": (v2 or {}).get("cross_summary"),
        "v3_summaries": (v3 or {}).get("summaries"),
        "v3_params": {"t_low": (v3 or {}).get("t_low"), "t_mid": (v3 or {}).get("t_mid")},
        "v5_champion_mean_acc": (v5 or {}).get("champion_mean_acc"),
        "v5_passes_95": (v5 or {}).get("passes_95"),
        "deploy_recommendation": {
            "prosqa_primary": "confidence_fallback τ=0.48（95.23%）",
            "cross_dataset": "hop4_tri_zone（V3 最优 T_low/T_mid + hop≥4）",
            "insight": "collateral 来自 τ 近邻误触发；tri-zone 软锁 + agreement",
        },
        "recommended_cross_summary": hop4_summary,
        "missing": [x for x, v in [
            ("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4), ("v5", v5),
        ] if v is None],
        "ok": len([x for x, v in [
            ("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4), ("v5", v5),
        ] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase34_result("v6_final_rollup", payload)

    md = PHASE34_OUT / "PHASE34_GPU_SUMMARY.md"
    lines = [
        "# Phase 34 · GPU 汇总\n",
        f"> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n",
    ]
    if v2:
        lines.append(f"## V2 best val\n- {json.dumps(v2.get('best_val'), ensure_ascii=False)}\n\n")
    if hop4_summary:
        lines.append(
            f"## V3 hop4_tri_zone cross\n"
            f"- weighted Δ: {hop4_summary.get('weighted_mean_delta_pp')} pp\n"
            f"- in-dist: {hop4_summary.get('in_dist_weighted_delta_pp')} pp\n"
            f"- OOD: {hop4_summary.get('ood_weighted_delta_pp')} pp\n\n"
        )
    if v5:
        lines.append(f"## V5 full 419\n- mean acc: {v5.get('champion_mean_acc')} passes_95={v5.get('passes_95')}\n")
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
