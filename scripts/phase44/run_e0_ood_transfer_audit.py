#!/usr/bin/env python3
"""E0 · 从 Phase32 审计「全新数据集代理」上置信度通解是否有效（无需 GPU）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase44_common import (  # noqa: E402
    OOD_PROXY_SLICES,
    TRANSFER_THR,
    load_phase32_t1,
    rollup_transfer_rows,
    slice_rows_by_ids,
    write_phase44_result,
)


def main() -> None:
    t1 = load_phase32_t1()
    if not t1:
        raise SystemExit("缺少 results/phase32/t1_cross_dataset_transfer_latest.json")

    all_rows = t1.get("slices") or []
    ood_rows = slice_rows_by_ids(t1, OOD_PROXY_SLICES)
    standard_rows = [r for r in all_rows if r.get("category") == "standard"]
    synthetic_rows = [r for r in all_rows if (r.get("slice_id") or "").startswith("syn_")]

    ood_roll = rollup_transfer_rows(ood_rows)
    std_roll = rollup_transfer_rows(standard_rows)
    syn_roll = rollup_transfer_rows(synthetic_rows)
    full_roll = t1.get("summary") or {}

    hurt_ood = [r for r in ood_rows if r.get("transfer_hurts")]
    helps_ood = [r for r in ood_rows if r.get("transfer_helps")]

    payload = {
        "experiment_id": "e0_ood_transfer_audit",
        "title": "E0 · 全新数据集代理审计（Phase32 冻结 τ=0.48）",
        "frozen_thr": TRANSFER_THR,
        "ood_proxy_slice_ids": OOD_PROXY_SLICES,
        "ood_proxy_rows": ood_rows,
        "ood_proxy_summary": ood_roll,
        "synthetic_only_summary": syn_roll,
        "standard_summary": std_roll,
        "full_phase32_summary": full_roll,
        "hurt_slices": [{"slice_id": r["slice_id"], "delta_pp": r["delta_pp"],
                         "fallback_rate": r.get("fallback_rate")} for r in hurt_ood],
        "help_slices": [{"slice_id": r["slice_id"], "delta_pp": r["delta_pp"],
                         "main_acc": r["main_acc"], "transfer_acc": r["transfer_acc"]}
                        for r in helps_ood],
        "conclusions": [
            f"OOD 代理 {len(ood_rows)} 切片：helps {ood_roll.get('transfer_helps_count', 0)} / "
            f"hurts {ood_roll.get('transfer_hurts_count', 0)}，mean Δ {ood_roll.get('mean_delta_pp')} pp",
            "纯合成链 syn_*：部分暴涨（syn_chain_6_wide +36pp），部分 hurt（syn_chain_5_wide）",
            "standard 子集：通解未必增益（hops_3 hurt）",
            "改进方向：hurt 切片做 τ 适配或禁用回退（见 E1/E2）",
        ],
        "ok": True,
    }
    path = write_phase44_result("e0_ood_transfer_audit", payload)
    print(f"Wrote {path}")
    print(payload["conclusions"][0])


if __name__ == "__main__":
    main()
