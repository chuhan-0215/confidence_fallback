"""Phase 43 · Terminal Lock（跨集终局 + v5 @99 增强档）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase37"))
sys.path.insert(0, str(ROOT / "scripts" / "phase42"))

from _phase37_common import dual_ok, m2_head_ready, unique_slice_ids  # noqa: E402
from _phase39_common import CANONICAL_SEED, DEFAULT_PROFILE, ROBUST_SEEDS  # noqa: E402
from _phase42_common import (  # noqa: E402
    WORST_SEED,
    eval_hybrid_v4_router,
    eval_hybrid_v5_router,
    router_v4_rules_doc,
    router_v5_rules_doc,
)

PANEL_SEEDS = (40, 41, 42, 43, 44, 45, 99, 100)


def write_phase43_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(43, eid, payload)


def load_phase42_d2() -> dict | None:
    import json
    for base in (ROOT / "results" / "phase42", ROOT / "outbox/results/from_a800/phase42"):
        p = base / "d2_hybrid_v5_seed_robust_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_phase42_d3() -> dict | None:
    import json
    for base in (ROOT / "results" / "phase42", ROOT / "outbox/results/from_a800/phase42"):
        p = base / "d3_deploy_bounds_v5_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


__all__ = [
    "CANONICAL_SEED",
    "DEFAULT_PROFILE",
    "PANEL_SEEDS",
    "ROBUST_SEEDS",
    "WORST_SEED",
    "dual_ok",
    "eval_hybrid_v4_router",
    "eval_hybrid_v5_router",
    "load_phase42_d2",
    "load_phase42_d3",
    "m2_head_ready",
    "router_v4_rules_doc",
    "router_v5_rules_doc",
    "unique_slice_ids",
    "write_phase43_result",
]
