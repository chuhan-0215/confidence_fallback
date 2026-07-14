"""Phase 35 · Unified Deploy（Agreement + Tri-zone 组合）。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(ROOT / "scripts" / "phase23"))

from _phase34_common import (  # noqa: E402
    CHAMPION_SEEDS,
    HURT_SLICE_IDS,
    PHASE34_OUT,
    T_LOW_GRID,
    T_MID_GRID,
    TRANSFER_THR,
    _full_knn,
    _knn_preview,
    _main_step,
    _policy_row,
    eval_agreement_lock,
    eval_tri_zone,
    m2_head_ready,
    split_val_test,
    unique_slice_ids,
)
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402

PHASE35_OUT = ROOT / "results" / "phase35"
P34_BEST = (0.40, 0.48)
ALT_GRID = ((0.38, 0.46), (0.35, 0.47), (0.40, 0.47))
MAIN_ACC_ROUTER_THR = 0.85
AGREEMENT_CONSTRUCTIONS = frozenset({"diamond", "prosqa_native", "prosqa_extend"})


def write_phase35_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(35, eid, payload)


def load_p34_best() -> tuple[float, float]:
    path = PHASE34_OUT / "v3_hop4_tri_zone_latest.json"
    if path.is_file():
        data = __import__("json").loads(path.read_text(encoding="utf-8"))
        return float(data.get("t_low", P34_BEST[0])), float(data.get("t_mid", P34_BEST[1]))
    v2 = PHASE34_OUT / "v2_tri_zone_sweep_latest.json"
    if v2.is_file():
        data = __import__("json").loads(v2.read_text(encoding="utf-8"))
        best = data.get("best_val") or {}
        return float(best.get("t_low", P34_BEST[0])), float(best.get("t_mid", P34_BEST[1]))
    return P34_BEST


@torch.no_grad()
def eval_agreement_tri_zone(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn,
    t_low: float, t_mid: float, hop4_only: bool = False,
):
    from boundary_budget import blind_depth

    head.eval()
    correct = fallback_count = agreement_skip = zone_mid = zone_low = 0
    for idx, sample in enumerate(samples):
        _, pred0, prob0, expected = _main_step(
            head, model, tokenizer, sample, device=device, seed=seed,
            profile=profile, struct_floor=struct_floor, idx=idx,
        )
        pred_prev = _knn_preview(
            model, tokenizer, sample, device=device, seed=seed,
            profile=profile, knn_floor=knn_floor, idx=idx,
        )
        do_fallback = False
        if pred0 == pred_prev:
            agreement_skip += 1
            final = pred0
        elif prob0 >= t_mid:
            final = pred0
        elif prob0 < t_low:
            do_fallback = True
            zone_low += 1
        else:
            do_fallback = True
            zone_mid += 1

        if do_fallback:
            if hop4_only and blind_depth(sample) < 4:
                final = pred0
            else:
                fallback_count += 1
                pk, _, _ = _full_knn(
                    head, model, tokenizer, sample, device=device, seed=seed,
                    profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn, idx=idx,
                )
                final = pk
        if final == expected:
            correct += 1
    return _policy_row(
        "agreement_tri_zone", correct, len(samples), fallback_count,
        {
            "t_low": t_low, "t_mid": t_mid, "hop4_only": hop4_only,
            "agreement_skip": agreement_skip,
            "zone_low_count": zone_low,
            "zone_mid_disagree_count": zone_mid,
        },
    )


@torch.no_grad()
def eval_main_acc_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn,
    t_low: float, t_mid: float, main_thr: float = MAIN_ACC_ROUTER_THR,
):
    """切片级路由：主路径 acc≥main_thr → agreement_lock，否则 combo。"""
    main_row = eval_main_path(
        model, tokenizer, samples, device=device, seed=seed,
        profile=profile, struct_floor=struct_floor,
    )
    if (main_row.get("accuracy") or 0.0) >= main_thr:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["main_thr"] = main_thr
        row["params"]["slice_main_acc"] = main_row["accuracy"]
        return row
    row = eval_agreement_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "agreement_tri_zone"
    row["params"]["main_thr"] = main_thr
    row["params"]["slice_main_acc"] = main_row["accuracy"]
    return row


@torch.no_grad()
def eval_construction_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn,
    t_low: float, t_mid: float, meta: dict,
):
    """可部署路由：高 collateral construction → agreement_lock，其余 combo。"""
    construction = (meta.get("construction") or "").lower()
    if construction in AGREEMENT_CONSTRUCTIONS:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["construction"] = construction
        return row
    row = eval_agreement_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "agreement_tri_zone"
    row["params"]["construction"] = construction
    return row
