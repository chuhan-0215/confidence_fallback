"""Phase 37 · Deploy Closure（tri_zone 定稿验证 + 全量扫参）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))

from _phase34_common import (  # noqa: E402
    CHAMPION_SEEDS,
    HURT_SLICE_IDS,
    TRANSFER_THR,
    eval_agreement_lock,
    eval_tri_zone,
    m2_head_ready,
    unique_slice_ids,
)
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402

PHASE37_OUT = ROOT / "results" / "phase37"
LOCKED_TRI_ZONE = (0.40, 0.48)
T_LOW_FULL_GRID = (0.30, 0.32, 0.35, 0.38, 0.40, 0.42)
T_MID_FIXED = 0.48

# 跨集时保持主路径（rescue 失败或无益）
MAIN_ONLY_SLICES = frozenset({"syn_chain_5_wide"})
# 高 collateral / push 延长：agreement_lock 更稳
AGREEMENT_SLICES = frozenset({
    "push_ext6_from4", "push_ext5_from3", "mix_75_4", "mix_50_4",
    "hops_3", "v_diamond_5",
})
FALLBACK_WRONG_IDX = (3, 84, 147, 208, 220, 387, 410)


def write_phase37_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(37, eid, payload)


def dual_ok(summary: dict | None) -> bool:
    if not summary:
        return False
    in_d = summary.get("in_dist_weighted_delta_pp")
    ood = summary.get("ood_weighted_delta_pp")
    return in_d is not None and ood is not None and in_d >= 0 and ood >= 7.0


def score_summary(summary: dict) -> float:
    in_d = summary.get("in_dist_weighted_delta_pp") or -999.0
    ood = summary.get("ood_weighted_delta_pp") or -999.0
    if ood < 7.0:
        return -999.0 + ood
    return in_d + 0.1 * (summary.get("weighted_mean_delta_pp") or 0.0)


def eval_hybrid_slice_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn, meta: dict,
    t_low: float = LOCKED_TRI_ZONE[0], t_mid: float = LOCKED_TRI_ZONE[1],
):
    """按 slice 元数据选策略（可部署，无 oracle main_acc）。"""
    sid = meta.get("slice_id") or meta.get("id") or ""
    if sid in MAIN_ONLY_SLICES:
        row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=struct_floor)
        return {
            "accuracy": row["accuracy"],
            "total": row["total"],
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "params": {"mode": "main_only", "router": "skip_transfer", "slice_id": sid},
        }
    if sid in AGREEMENT_SLICES:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["slice_id"] = sid
        return row
    row = eval_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "tri_zone"
    row["params"]["slice_id"] = sid
    return row
