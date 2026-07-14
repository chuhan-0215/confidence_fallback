"""Phase 25 · 终稿验证：Y2 精调 / 4跳专项回退 / deploy spec v2。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase24"))
sys.path.insert(0, str(ROOT / "scripts" / "phase23"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase4._phase4_common import timed_run, utc_now  # noqa: E402
from phase23._phase23_common import (  # noqa: E402
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_json, load_m2_head_state, load_rich_head, load_splits,
    stats, timing_metrics,
)
from boundary_budget import blind_depth  # noqa: E402

PHASE25_OUT = ROOT / "results" / "phase25"
ROBUST_SEEDS = (0, 1, 2, 42, 99)
FINE_FALLBACK = [0.42, 0.45, 0.48, 0.50, 0.52, 0.55]


def write_phase25_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(25, eid, payload)
