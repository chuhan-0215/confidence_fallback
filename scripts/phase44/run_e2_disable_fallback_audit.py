#!/usr/bin/env python3
"""E2 · 改进基线：hurt 切片禁用回退（τ→∞）vs 冻结 τ=0.48。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase44_common import load_phase32_t1, write_phase44_result  # noqa: E402


def main() -> None:
    t1 = load_phase32_t1()
    if not t1:
        raise SystemExit("缺少 Phase32 T1 结果")

    rows = []
    for r in t1.get("slices") or []:
        fb = r.get("fallback_rate") or 0
        delta = r.get("delta_pp") or 0
        main_acc = r.get("main_acc") or 0
        transfer_acc = r.get("transfer_acc") or 0
        # 禁用回退 = 永远用主路径 ≈ main_acc
        disable_gain = round((main_acc - transfer_acc) * 100, 3) if delta < 0 else 0
        rows.append({
            "slice_id": r["slice_id"],
            "category": r.get("category"),
            "delta_pp": delta,
            "fallback_rate": fb,
            "transfer_hurts": r.get("transfer_hurts"),
            "disable_fallback_would_help_pp": disable_gain,
            "recommendation": (
                "skip_transfer（禁用回退）" if delta < -0.5
                else ("keep_transfer" if delta > 0.5 else "neutral")
            ),
        })

    hurt_rows = [r for r in rows if r["transfer_hurts"]]
    skip_candidates = sorted(hurt_rows, key=lambda x: x["disable_fallback_would_help_pp"], reverse=True)

    payload = {
        "experiment_id": "e2_disable_fallback_audit",
        "title": "E2 · hurt 切片禁用回退审计",
        "hurt_count": len(hurt_rows),
        "skip_candidates": skip_candidates[:15],
        "rule": "若 transfer_hurts 且 disable_fallback_would_help>0 → 该数据集不要用置信度回退",
        "all_rows": rows,
        "ok": True,
    }
    path = write_phase44_result("e2_disable_fallback_audit", payload)
    print(f"Wrote {path}; hurt={len(hurt_rows)}")


if __name__ == "__main__":
    main()
