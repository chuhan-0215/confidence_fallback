"""Phase 40 · Deploy Closure（worst-seed 审计 + hybrid v3）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))
sys.path.insert(0, str(ROOT / "scripts" / "phase39"))

from eval_profile import parse_eval_profile  # noqa: E402
from _phase37_common import (  # noqa: E402
    LOCKED_TRI_ZONE,
    MAIN_ONLY_SLICES,
    dual_ok,
    m2_head_ready,
    unique_slice_ids,
)
from _phase39_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    HYBRID_V2_AGREEMENT,
    ROBUST_SEEDS,
    eval_hybrid_v2_router,
)
from _phase34_common import TRANSFER_THR, eval_agreement_lock, eval_tri_zone  # noqa: E402
from shared.eval_paths import eval_main_path  # noqa: E402

WORST_SEED = 43
STRESS_SEEDS = (42, 43, 44, 99)

# v3: v_diamond_5 在 seed43 翻转 -8pp（P38 Z2）→ skip_transfer
HYBRID_V3_EXTRA_SKIP = frozenset({"v_diamond_5"})


def write_phase40_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(40, eid, payload)


def eval_hybrid_v3_router(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn, meta: dict,
    t_low: float = LOCKED_TRI_ZONE[0], t_mid: float = LOCKED_TRI_ZONE[1],
):
    sid = meta.get("slice_id") or meta.get("id") or ""
    skip = MAIN_ONLY_SLICES | HYBRID_V3_EXTRA_SKIP
    if sid in skip:
        row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=struct_floor)
        return {
            "accuracy": row["accuracy"], "total": row["total"],
            "fallback_count": 0, "fallback_rate": 0.0,
            "params": {"mode": "main_only", "router": "skip_transfer", "slice_id": sid, "variant": "v3"},
        }
    if sid in HYBRID_V2_AGREEMENT:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["variant"] = "v3"
        return row
    row = eval_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "tri_zone"
    row["params"]["variant"] = "v3"
    return row


def router_v3_rules_doc() -> dict:
    return {
        "skip_transfer": sorted(MAIN_ONLY_SLICES | HYBRID_V3_EXTRA_SKIP),
        "agreement_lock": sorted(HYBRID_V2_AGREEMENT),
        "tri_zone_default": "all other slices (incl. hops_3)",
        "params": {"t_low": LOCKED_TRI_ZONE[0], "t_mid": LOCKED_TRI_ZONE[1]},
        "changes_vs_v5": ["v_diamond_5 → skip_transfer（seed43 翻转修复）"],
    }


__all__ = [
    "CANONICAL_SEED",
    "DEFAULT_PROFILE",
    "ROBUST_SEEDS",
    "STRESS_SEEDS",
    "WORST_SEED",
    "dual_ok",
    "eval_hybrid_v2_router",
    "eval_hybrid_v3_router",
    "m2_head_ready",
    "router_v3_rules_doc",
    "unique_slice_ids",
    "write_phase40_result",
]
