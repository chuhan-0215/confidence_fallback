"""Phase 41 · V3 Lock & OOD Push（锁定 v3 + push_ext7 skip 候选）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase40"))

from _phase37_common import dual_ok, m2_head_ready, unique_slice_ids  # noqa: E402
from _phase39_common import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_PROFILE,
    HYBRID_V2_AGREEMENT,
    ROBUST_SEEDS,
    eval_hybrid_v2_router,
)
from _phase40_common import (  # noqa: E402
    HYBRID_V3_EXTRA_SKIP,
    STRESS_SEEDS,
    WORST_SEED,
    eval_hybrid_v3_router,
    router_v3_rules_doc,
)

# v4: v3 + push_ext7 系列 skip（B1 seed43 −6/−2pp，@99 均为 0pp）
HYBRID_V4_EXTRA_SKIP = HYBRID_V3_EXTRA_SKIP | frozenset({"push_ext7_from3", "push_ext7_mixed"})
HYBRID_V4_AGREEMENT = HYBRID_V2_AGREEMENT - HYBRID_V4_EXTRA_SKIP
OOD_SEEDS = (42, 44)


def write_phase41_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(41, eid, payload)


def eval_hybrid_v4_router(
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
    skip = MAIN_ONLY_SLICES | HYBRID_V4_EXTRA_SKIP
    if sid in skip:
        row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=struct_floor)
        return {
            "accuracy": row["accuracy"], "total": row["total"],
            "fallback_count": 0, "fallback_rate": 0.0,
            "params": {"mode": "main_only", "router": "skip_transfer", "slice_id": sid, "variant": "v4"},
        }
    if sid in HYBRID_V4_AGREEMENT:
        row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        row["params"]["router"] = "agreement_lock"
        row["params"]["variant"] = "v4"
        return row
    row = eval_tri_zone(
        head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        t_low=t_low, t_mid=t_mid, hop4_only=False,
    )
    row["params"]["router"] = "tri_zone"
    row["params"]["variant"] = "v4"
    return row


def router_v4_rules_doc() -> dict:
    from _phase37_common import LOCKED_TRI_ZONE, MAIN_ONLY_SLICES
    return {
        "skip_transfer": sorted(MAIN_ONLY_SLICES | HYBRID_V4_EXTRA_SKIP),
        "agreement_lock": sorted(HYBRID_V4_AGREEMENT),
        "tri_zone_default": "all other slices (incl. hops_3)",
        "params": {"t_low": LOCKED_TRI_ZONE[0], "t_mid": LOCKED_TRI_ZONE[1]},
        "changes_vs_v3": ["push_ext7_from3/mixed → skip_transfer（B1 seed43 翻转）"],
    }


def load_phase40_b3() -> dict | None:
    import json
    for base in (ROOT / "results" / "phase40", ROOT / "outbox/results/from_a800/phase40"):
        p = base / "b3_deploy_bounds_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def pick_deploy_variant(b3: dict | None, b2_c2: dict | None) -> tuple[str, str]:
    """Return (policy_name, variant). Prefer v4 if more dual_ok; else v3 if B3 strict ok."""
    v3_ok = (b2_c2 or {}).get("v3_dual_ok_count", 0)
    v4_ok = (b2_c2 or {}).get("v4_dual_ok_count", 0)
    if v4_ok > v3_ok:
        return "hybrid_slice_router_v4", "v4"
    bounds = (b3 or {}).get("deploy_bounds") or {}
    v3_strict = ((bounds.get("hybrid_v3") or {}).get("deploy_ok_strict"))
    if v3_strict:
        return "hybrid_slice_router_v3", "v3"
    return "hybrid_slice_router_v2", "v5"


__all__ = [
    "CANONICAL_SEED",
    "DEFAULT_PROFILE",
    "OOD_SEEDS",
    "ROBUST_SEEDS",
    "STRESS_SEEDS",
    "WORST_SEED",
    "dual_ok",
    "eval_hybrid_v2_router",
    "eval_hybrid_v3_router",
    "eval_hybrid_v4_router",
    "load_phase40_b3",
    "m2_head_ready",
    "pick_deploy_variant",
    "router_v3_rules_doc",
    "router_v4_rules_doc",
    "unique_slice_ids",
    "write_phase41_result",
]
