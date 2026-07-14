#!/usr/bin/env python3
"""X6 · Phase 36 终局 rollup + deploy_spec_v3。"""
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


def pick_best(summaries: list[tuple[str, dict | None]]) -> dict | None:
    best = None
    best_score = -999.0
    for name, s in summaries:
        if not s:
            continue
        in_d = s.get("in_dist_weighted_delta_pp") or -999.0
        ood = s.get("ood_weighted_delta_pp") or -999.0
        dual_ok = in_d >= 0 and ood >= 7.0
        score = (100.0 if dual_ok else 0.0) + in_d + 0.05 * ood
        if score > best_score:
            best_score = score
            best = {"name": name, "score": round(score, 3), "dual_ok": dual_ok, "summary": s}
    return best


def main() -> None:
    from _phase36_common import write_phase36_result

    x1 = load_json(36, "x1_w2_winner_full_slices")
    x2 = load_json(36, "x2_three_way_shootout")
    x3 = load_json(36, "x3_push_ext6_forensic")
    x4 = load_json(36, "x4_baseline_gap_audit")
    x5 = load_json(36, "x5_full419_validate")
    p35 = load_json(35, "w6_final_rollup")

    x1s = (x1 or {}).get("summaries") or {}
    x2s = (x2 or {}).get("summaries") or {}
    best_cross = pick_best([
        ("combo_w2_full", x1s.get("combo_w2")),
        ("combo_p34_full", x1s.get("combo_p34")),
        ("tri_zone", x2s.get("tri_zone")),
        ("combo_w2", x2s.get("combo_w2")),
        ("category_router", x2s.get("category_router")),
        ("p35_w2_test", ((p35 or {}).get("w2_cross_summary"))),
        ("p34_tri_zone", ((p35 or {}).get("phase34_reference") or {}).get("tri_zone")),
    ])

    deploy = {
        "prosqa_primary": "confidence_fallback τ=0.48",
        "cross_dataset": (best_cross or {}).get("name", "combo_w2"),
        "dual_deploy_note": "ProsQA 内用 baseline；跨未知分布用 combo/category 路由",
    }
    if best_cross and best_cross.get("dual_ok"):
        deploy["cross_dataset_locked"] = best_cross["name"]

    payload = {
        "experiment_id": "x6_final_rollup",
        "title": "X6 · Phase 36 终局汇总",
        "phase35_w2_test_summary": (p35 or {}).get("w2_cross_summary"),
        "x1_summaries": x1s,
        "x2_summaries": x2s,
        "x4_baseline_audit": {
            "p25_reference": (x4 or {}).get("p25_reference_acc"),
            "current_acc": (x4 or {}).get("current_batch_acc"),
            "wrong_count": (x4 or {}).get("wrong_count"),
            "fallback_wrong_count": (x4 or {}).get("fallback_wrong_count"),
        },
        "x5_full419": {
            "champion": (x5 or {}).get("champion"),
            "champion_acc": (x5 or {}).get("champion_acc"),
            "policies": (x5 or {}).get("policies"),
        },
        "best_cross_policy": best_cross,
        "deploy_spec_v3": deploy,
        "missing": [x for x, v in [
            ("x1", x1), ("x2", x2), ("x3", x3), ("x4", x4), ("x5", x5),
        ] if v is None],
        "ok": len([x for x, v in [
            ("x1", x1), ("x2", x2), ("x3", x3), ("x4", x4), ("x5", x5),
        ] if v is None]) == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_phase36_result("x6_final_rollup", payload)

    md = ROOT / "results" / "phase36" / "PHASE36_GPU_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Phase 36 · GPU 汇总\n> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    if best_cross:
        s = best_cross["summary"]
        lines.append(
            f"## 最优跨集: {best_cross['name']} (dual_ok={best_cross['dual_ok']})\n"
            f"- in-dist: {s.get('in_dist_weighted_delta_pp')} pp\n"
            f"- OOD: {s.get('ood_weighted_delta_pp')} pp\n"
            f"- weighted Δ: {s.get('weighted_mean_delta_pp')} pp\n\n"
        )
    if x5:
        lines.append(f"## X5 full 419\n- champion: {x5.get('champion')} acc={x5.get('champion_acc')}\n")
    if x4:
        lines.append(
            f"## X4 baseline audit\n"
            f"- current: {x4.get('current_batch_acc')} vs P25 {x4.get('p25_reference_acc')}\n"
            f"- fallback wrong: {x4.get('fallback_wrong_count')}\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
