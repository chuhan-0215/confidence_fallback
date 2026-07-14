#!/usr/bin/env python3
"""W6 · Phase 35 终局 rollup。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase35_common import PHASE35_OUT, write_phase35_result  # noqa: E402


def load_json(name: str) -> dict | None:
    path = PHASE35_OUT / f"{name}_latest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pick_best_cross(*summaries: tuple[str, dict | None]) -> dict | None:
    best = None
    best_score = -999.0
    for name, summary in summaries:
        if not summary:
            continue
        in_d = summary.get("in_dist_weighted_delta_pp") or -999.0
        ood = summary.get("ood_weighted_delta_pp") or -999.0
        score = in_d + (0.05 * ood if ood >= 7.0 else ood - 20.0)
        if score > best_score:
            best_score = score
            best = {"name": name, "score": round(score, 3), "summary": summary}
    return best


def main() -> None:
    w1 = load_json("w1_combo_cross_transfer")
    w2 = load_json("w2_combo_grid_sweep")
    w3 = load_json("w3_adaptive_router")
    w4 = load_json("w4_collateral_replay")
    w5 = load_json("w5_champion_validate")
    p34 = None
    p34_path = ROOT / "outbox/results/from_a800/phase34/v6_final_rollup_latest.json"
    if p34_path.is_file():
        p34 = json.loads(p34_path.read_text(encoding="utf-8"))

    w1_summaries = (w1 or {}).get("summaries") or {}
    w3_summaries = (w3 or {}).get("summaries") or {}
    best_cross = pick_best_cross(
        ("combo", w1_summaries.get("combo")),
        ("combo_hop4", w1_summaries.get("combo_hop4")),
        ("w2_combo_best", (w2 or {}).get("cross_summary")),
        ("main_acc_router", w3_summaries.get("main_acc_router")),
        ("construction_router", w3_summaries.get("construction_router")),
        ("p34_agreement", ((p34 or {}).get("v1_summaries") or {}).get("agreement_lock")),
        ("p34_tri_zone", ((p34 or {}).get("v3_summaries") or {}).get("tri_zone")),
    )

    payload = {
        "experiment_id": "w6_final_rollup",
        "title": "W6 · Phase 35 终局汇总",
        "phase34_reference": {
            "agreement_lock": ((p34 or {}).get("v1_summaries") or {}).get("agreement_lock"),
            "tri_zone": ((p34 or {}).get("v3_summaries") or {}).get("tri_zone"),
            "hop4_tri_zone": ((p34 or {}).get("v3_summaries") or {}).get("hop4_tri_zone"),
        },
        "w1_summaries": w1_summaries,
        "w2_best": (w2 or {}).get("best_val"),
        "w2_cross_summary": (w2 or {}).get("cross_summary"),
        "w2_top5": (w2 or {}).get("top5_cross"),
        "w3_summaries": w3_summaries,
        "w5_champion": (w5 or {}).get("champion"),
        "w5_champion_acc": (w5 or {}).get("champion_acc"),
        "w5_baseline_acc": (w5 or {}).get("baseline_acc"),
        "w5_passes_95": (w5 or {}).get("passes_95"),
        "best_cross_policy": best_cross,
        "deploy_recommendation": {
            "prosqa_primary": "confidence_fallback τ=0.48（P25 定稿 95.23%）",
            "cross_dataset": (best_cross or {}).get("name", "agreement_tri_zone"),
            "insight": "P34 分离验证 agreement(in-dist+) 与 tri-zone(OOD+)；P35 组合二者",
        },
        "missing": [x for x, v in [
            ("w1", w1), ("w2", w2), ("w3", w3), ("w4", w4), ("w5", w5),
        ] if v is None],
        "ok": len([x for x, v in [
            ("w1", w1), ("w2", w2), ("w3", w3), ("w4", w4), ("w5", w5),
        ] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase35_result("w6_final_rollup", payload)

    md = PHASE35_OUT / "PHASE35_GPU_SUMMARY.md"
    lines = [
        "# Phase 35 · GPU 汇总\n",
        f"> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n",
    ]
    if best_cross:
        s = best_cross["summary"]
        lines.append(
            f"## 最优跨集策略: {best_cross['name']}\n"
            f"- weighted Δ: {s.get('weighted_mean_delta_pp')} pp\n"
            f"- in-dist: {s.get('in_dist_weighted_delta_pp')} pp\n"
            f"- OOD: {s.get('ood_weighted_delta_pp')} pp\n"
            f"- hurts: {s.get('hurts_count')}\n\n"
        )
    if w5:
        lines.append(
            f"## W5 full 419 (seed=99)\n"
            f"- champion: {w5.get('champion')} acc={w5.get('champion_acc')}\n"
            f"- baseline: {w5.get('baseline_acc')} passes_95={w5.get('passes_95')}\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
