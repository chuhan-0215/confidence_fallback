"""Phase 38 · Robust Lock（hybrid_router seed 审计 + deploy_spec_v4）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))

from _phase37_common import (  # noqa: E402
    LOCKED_TRI_ZONE,
    dual_ok,
    eval_hybrid_slice_router,
    m2_head_ready,
    score_summary,
    unique_slice_ids,
)
from _phase34_common import eval_tri_zone  # noqa: E402

PHASE38_OUT = ROOT / "results" / "phase38"
ROBUST_SEEDS = (42, 43, 44, 99)
CANONICAL_SEED = 99


def write_phase38_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(38, eid, payload)


def router_rules_doc() -> dict:
    from _phase37_common import AGREEMENT_SLICES, MAIN_ONLY_SLICES
    return {
        "skip_transfer": sorted(MAIN_ONLY_SLICES),
        "agreement_lock": sorted(AGREEMENT_SLICES),
        "tri_zone_default": "all other slices",
        "params": {"t_low": LOCKED_TRI_ZONE[0], "t_mid": LOCKED_TRI_ZONE[1]},
    }
