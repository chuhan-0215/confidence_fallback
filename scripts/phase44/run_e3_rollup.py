#!/usr/bin/env python3
"""E3 · Phase44 汇总：零样本结论 + 改进建议。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase44_common import write_phase44_result  # noqa: E402

PHASE44 = ROOT / "results" / "phase44"


def _load(name: str) -> dict | None:
    p = PHASE44 / f"{name}_latest.json"
    if not p.is_file():
        p = PHASE44 / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> None:
    e0 = _load("e0_ood_transfer_audit")
    e1 = _load("e1_tau_adaptation")
    e2 = _load("e2_disable_fallback_audit")

    ood = (e0 or {}).get("ood_proxy_summary") or {}
    rescued = (e1 or {}).get("rescued_slices") or []
    hurt = (e2 or {}).get("hurt_count")

    recommendations = []
    if ood.get("transfer_hurts_count", 0) > 0:
        recommendations.append("换到 OOD 数据集：先小样本试点，hurt 则禁用回退或重标定 τ")
    if rescued:
        recommendations.append(f"τ 适配可救回切片：{rescued}")
    if hurt:
        recommendations.append(f"全库 {hurt} 个 hurt 切片应列入 skip 表，不触发 confidence_fallback")
    if not recommendations:
        recommendations.append("零样本通解在 OOD 代理上净收益为正")

    payload = {
        "experiment_id": "e3_external_transfer_rollup",
        "title": "E3 · 置信度通解外推汇总",
        "e0_ood_mean_delta_pp": ood.get("mean_delta_pp"),
        "e0_helps_hurts": f"{ood.get('transfer_helps_count')}/{ood.get('transfer_hurts_count')}",
        "e1_rescued_slices": rescued,
        "e2_hurt_count": hurt,
        "recommendations": recommendations,
        "improvement_menu": [
            "1. 试点：50 题测 frozen τ=0.48 的 Δ",
            "2. hurt → 禁用回退（等价 τ=0.99 / skip_transfer）",
            "3. 边缘 → 在该集 dev 上扫 τ（E1）",
            "4. 仍不行 → 目标域重训 M2 Stop Head（未在本 Phase 跑）",
        ],
        "missing": [x for x, d in [("e0", e0), ("e1", e1), ("e2", e2)] if not d],
        "ok": bool(e0),
    }
    path = write_phase44_result("e3_external_transfer_rollup", payload)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
