"""Phase 39 · Seed-Stable Cross（固定 profile + hybrid v2）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))

from eval_profile import parse_eval_profile  # noqa: E402
from _phase37_common import (  # noqa: E402
    AGREEMENT_SLICES,
    LOCKED_TRI_ZONE,
    MAIN_ONLY_SLICES,
    dual_ok,
    eval_hybrid_slice_router,
    m2_head_ready,
    unique_slice_ids,
)
from _phase34_common import TRANSFER_THR, eval_agreement_lock, eval_tri_zone  # noqa: E402
from shared.eval_paths import eval_main_path  # noqa: E402

PHASE39_OUT = ROOT / "results" / "phase39"
ROBUST_SEEDS = (42, 43, 44, 99)
CANONICAL_SEED = 99

FIXED_PROFILE = parse_eval_profile({
    "prompt_mode": "fixed_edges",
    "choice_order": "target_first",
})
DEFAULT_PROFILE = parse_eval_profile(None)

# v2: hops_3 改 tri_zone（P38 Z3 tri +0.5pp）；补 push_ext7 seed43 hurt
HYBRID_V2_AGREEMENT = frozenset(
    (AGREEMENT_SLICES - {"hops_3"}) | {"push_ext7_from3", "push_ext7_mixed"}
)


def write_phase39_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(39, eid, payload)


def eval_hybrid_v2_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn, meta: dict,
    t_low: float = LOCKED_TRI_ZONE[0], t_mid: float = LOCKED_TRI_ZONE[1],
):
    sid = meta.get("slice_id") or meta.get("id") or ""
    if sid in MAIN_ONLY_SLICES:
        row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=struct_floor)
        return {
            "accuracy": row["accuracy"], "total": row["total"],
            "fallback_count": 0, "fallback_rate": 0.0,
            "params": {"mode": "main_only", "router": "skip_transfer", "slice_id": sid, "variant": "v2"},
        }
    if sid in HYBRID_V2_AGREEMENT:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["variant"] = "v2"
        return row
    row = eval_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "tri_zone"
    row["params"]["variant"] = "v2"
    return row


# Re-export for phase39 runners
__all__ = [
    "CANONICAL_SEED",
    "DEFAULT_PROFILE",
    "FIXED_PROFILE",
    "ROBUST_SEEDS",
    "dual_ok",
    "eval_hybrid_v2_router",
    "m2_head_ready",
    "router_v2_rules_doc",
    "unique_slice_ids",
    "write_phase39_result",
]


def router_v2_rules_doc() -> dict:
    return {
        "skip_transfer": sorted(MAIN_ONLY_SLICES),
        "agreement_lock": sorted(HYBRID_V2_AGREEMENT),
        "tri_zone_default": "all other slices (incl. hops_3)",
        "params": {"t_low": LOCKED_TRI_ZONE[0], "t_mid": LOCKED_TRI_ZONE[1]},
        "changes_vs_v4": ["hops_3 → tri_zone", "push_ext7_from3/mixed → agreement"],
    }
