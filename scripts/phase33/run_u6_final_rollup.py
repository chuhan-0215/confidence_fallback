#!/usr/bin/env python3
"""U6 · Phase 33 终局 rollup。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase33_common import PHASE33_OUT, write_phase33_result  # noqa: E402


def load_json(name: str) -> dict | None:
    path = PHASE33_OUT / f"{name}_latest.json"
    if not path.is_file():
        path = ROOT / "outbox" / "results" / "from_a800" / "phase32" / f"{name}_latest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    u1 = load_json("u1_variant_cross_transfer")
    u2 = load_json("u2_surgical_cross_transfer")
    u3 = load_json("u3_category_tau_calibrate")
    u4 = load_json("u4_hurt_taxonomy")
    u5 = load_json("u5_unified_policy")
    p32 = load_json("t1_cross_dataset_transfer")

    best_variant = None
    if u1:
        best_variant = max(u1.get("summaries", {}).items(), key=lambda kv: kv[1].get("weighted_mean_delta_pp") or -999)

    payload = {
        "experiment_id": "u6_final_rollup",
        "title": "U6 · Phase 33 终局汇总",
        "phase32_baseline": (p32 or {}).get("summary"),
        "u1_best_variant": {"name": best_variant[0], "summary": best_variant[1]} if best_variant else None,
        "u2_surgical_summary": (u2 or {}).get("summary"),
        "u3_pooled_test": (u3 or {}).get("pooled_test"),
        "u3_global_optimal_tau": (u3 or {}).get("global_optimal", {}).get("best_threshold"),
        "u4_aggregate_collateral": (u4 or {}).get("aggregate_collateral_rate"),
        "u5_recommended_policy": (u5 or {}).get("recommended_policy"),
        "u5_full_419": (u5 or {}).get("full_419"),
        "u5_cross_summaries": (u5 or {}).get("cross_summaries"),
        "deploy_recommendation": {
            "prosqa_primary": "confidence_fallback τ=0.48（P25 acc champion 95.23%）",
            "cross_dataset": (u5 or {}).get("recommended_policy") or "S2_surgical or S3_dual_threshold",
            "insight": "通解=rescue mechanism；跨集需 hop 分区/死区/仲裁减 collateral",
        },
        "missing": [x for x, v in [
            ("u1", u1), ("u2", u2), ("u3", u3), ("u4", u4), ("u5", u5),
        ] if v is None],
    }
    write_phase33_result("u6_final_rollup", payload)

    summary_md = PHASE33_OUT / "PHASE33_GPU_SUMMARY.md"
    lines = [
        "# Phase 33 · GPU 汇总\n",
        f"- recommended: {(u5 or {}).get('recommended_policy', 'pending')}\n",
        f"- u4 collateral rate: {(u4 or {}).get('aggregate_collateral_rate', 'pending')}\n",
        f"- u3 global optimal τ: {(u3 or {}).get('global_optimal', {}).get('best_threshold', 'pending')}\n",
    ]
    if u5 and u5.get("full_419"):
        lines.append("- full 419:\n")
        for k, v in u5["full_419"].items():
            lines.append(f"  - {k}: {v.get('accuracy', 0):.1%}\n")
    summary_md.write_text("".join(lines), encoding="utf-8")
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
