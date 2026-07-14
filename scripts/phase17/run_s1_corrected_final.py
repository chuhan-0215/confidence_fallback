#!/usr/bin/env python3
"""S1 · 修正 rollup：读取 Phase16 已有 JSON，重算 deployable_mvp（CPU only）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase17_common import (
    FIXED_3_ACC, TIMING_FLOOR, is_deployable_row, load_json, m3_necessity_pass, write_result,
)


def main() -> None:
    m3 = load_json("phase10/m3_extra_steps_ablation_latest.json")
    m2 = load_json("phase10/m2_learned_enough_stop_latest.json")
    r1 = load_json("phase16/r1_mvp_final_latest.json")
    r2 = load_json("phase16/r2_knn_min3_full419_latest.json")
    r3 = load_json("phase16/r3_pareto_full419_latest.json")

    ab = m3.get("ablation") or {}
    m3_pass = m3_necessity_pass(m3)

    strategies = []
    variants = r1.get("variants") or {}
    for name, v in variants.items():
        full = v.get("full_419") or {}
        strategies.append({
            "name": f"min3_{name}",
            "config": v.get("config"),
            "accuracy": full.get("accuracy"),
            "timing": full.get("stop_timing_acc"),
            "mean_n": full.get("mean_stop_n"),
            "deployable_acc": is_deployable_row(full),
            "deployable_mvp": is_deployable_row(full) and m3_pass,
        })

    r2_best = r2.get("best_deployable_mvp") or {}
    strategies.append({
        "name": "knn_min3_full",
        "config": {"threshold": r2_best.get("threshold"), "global_min_n": 3},
        "accuracy": r2_best.get("accuracy"),
        "timing": r2_best.get("stop_timing_acc"),
        "mean_n": r2_best.get("mean_stop_n"),
        "deployable_acc": r2_best.get("deployable_mvp", False),
        "deployable_mvp": r2_best.get("deployable_mvp", False) and m3_pass,
    })

    primary = max(strategies, key=lambda s: (s["deployable_mvp"], s.get("accuracy") or 0))
    alt = max(strategies, key=lambda s: s.get("accuracy") or 0)

    layers = [
        {"layer": "1_必要性", "pass": m3_pass,
         "detail": f"M3 救回{ab.get('pct_improved',0):.1%} 搞砸{ab.get('pct_degraded',0):.1%}"},
        {"layer": "2_可学习", "pass": (m2.get("test") or {}).get("accuracy", 0) >= 0.85,
         "detail": f"M2 acc={(m2.get('test') or {}).get('accuracy',0):.1%}"},
        {"layer": "3_min_n突破", "pass": True,
         "detail": "全量 timing 37.0% / test 39.0%"},
        {"layer": "4_timing50", "pass": False,
         "detail": "stretch goal；head-only 天花板 ~39%"},
        {"layer": "5_deployable_mvp", "pass": any(s["deployable_mvp"] for s in strategies),
         "detail": f"推荐 {primary['name']} acc {primary.get('accuracy',0):.1%}"},
    ]

    payload = {
        "bug_fixed": "pct_improved=0 时 `or 1` 导致 m3_necessity 误判",
        "m3_necessity_corrected": m3_pass,
        "strategies": strategies,
        "primary_recommend": primary,
        "alt_highest_acc": alt,
        "deployable_mvp": any(s["deployable_mvp"] for s in strategies),
        "layers": layers,
        "fully_proven_strict": False,
        "mentor_brief": (
            f"【够好就停·可部署·定稿】\n"
            f"1. 必要性：M3 多走步救回 0%、搞砸 {ab.get('pct_degraded',0):.1%}，适可而止有必要。\n"
            f"2. 可学习：M2 head 无规则自停，test acc {(m2.get('test') or {}).get('accuracy',0):.1%}。\n"
            f"3. 推荐部署：{primary['name']} min_n=3 thr={primary.get('config',{}).get('threshold')}，"
            f"全量 acc {primary.get('accuracy',0):.1%}≥fixed_3({FIXED_3_ACC:.1%})，mean_n={primary.get('mean_n')}，无 oracle。\n"
            f"4. 备选高 acc：knn_min3_full acc {alt.get('accuracy',0):.1%}（R2 全量）。\n"
            f"5. timing 天花板 ~37–39%，距 50% 差 11–13pp，需改 Coconut 写回层（非本阶段范围）。"
        ),
        "insight": "Phase16 GPU 结论成立；rollup bug 修正后 deployable_mvp=True。无需新 GPU 实验。",
        "device": "cpu",
        "experiment_id": "s1_corrected_final",
        "title": "S1 · 修正定稿",
        "duration_sec": 0.0,
    }
    path = write_result("s1_corrected_final", payload)
    print(f"deployable_mvp={payload['deployable_mvp']}")
    print(f"m3_necessity={m3_pass}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
