"""Phase 42 · Seed43 Closure（v4 3/4 → 冲击 4/4）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase41"))

from _phase37_common import dual_ok, m2_head_ready, unique_slice_ids  # noqa: E402
from _phase39_common import CANONICAL_SEED, DEFAULT_PROFILE, ROBUST_SEEDS  # noqa: E402
from _phase41_common import (  # noqa: E402
    HYBRID_V4_AGREEMENT,
    HYBRID_V4_EXTRA_SKIP,
    WORST_SEED,
    eval_hybrid_v4_router,
    router_v4_rules_doc,
)

# v5: v4 + push_ext5_from3 skip（P40 B1 agreement @99 −2pp 且 seed 间翻转）
HYBRID_V5_EXTRA_SKIP = HYBRID_V4_EXTRA_SKIP | frozenset({"push_ext5_from3"})
HYBRID_V5_AGREEMENT = HYBRID_V4_AGREEMENT - frozenset({"push_ext5_from3"})

IN_DIST_CATS = frozenset({"standard", "pattern"})


def write_phase42_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(42, eid, payload)


def eval_hybrid_v5_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn, meta: dict,
    t_low: float = 0.4, t_mid: float = 0.48,
):
    from _phase37_common import LOCKED_TRI_ZONE, MAIN_ONLY_SLICES
    from _phase34_common import TRANSFER_THR, eval_agreement_lock, eval_tri_zone
    from shared.eval_paths import eval_main_path

    t_low = LOCKED_TRI_ZONE[0] if t_low == 0.4 else t_low
    t_mid = LOCKED_TRI_ZONE[1] if t_mid == 0.48 else t_mid
    sid = meta.get("slice_id") or meta.get("id") or ""
    skip = MAIN_ONLY_SLICES | HYBRID_V5_EXTRA_SKIP
    if sid in skip:
        row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=struct_floor)
        return {
            "accuracy": row["accuracy"], "total": row["total"],
            "fallback_count": 0, "fallback_rate": 0.0,
            "params": {"mode": "main_only", "router": "skip_transfer", "slice_id": sid, "variant": "v5"},
        }
    if sid in HYBRID_V5_AGREEMENT:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["variant"] = "v5"
        return row
    row = eval_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "tri_zone"
    row["params"]["variant"] = "v5"
    return row


def router_v5_rules_doc() -> dict:
    from _phase37_common import LOCKED_TRI_ZONE, MAIN_ONLY_SLICES
    return {
        "skip_transfer": sorted(MAIN_ONLY_SLICES | HYBRID_V5_EXTRA_SKIP),
        "agreement_lock": sorted(HYBRID_V5_AGREEMENT),
        "tri_zone_default": "all other slices (incl. hops_3)",
        "params": {"t_low": LOCKED_TRI_ZONE[0], "t_mid": LOCKED_TRI_ZONE[1]},
        "changes_vs_v4": ["push_ext5_from3 → skip_transfer（seed43 in-dist 修复候选）"],
    }


__all__ = [
    "CANONICAL_SEED",
    "DEFAULT_PROFILE",
    "IN_DIST_CATS",
    "ROBUST_SEEDS",
    "WORST_SEED",
    "dual_ok",
    "eval_hybrid_v4_router",
    "eval_hybrid_v5_router",
    "m2_head_ready",
    "router_v4_rules_doc",
    "router_v5_rules_doc",
    "unique_slice_ids",
    "write_phase42_result",
]
