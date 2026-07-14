"""Phase 24 · 终局收官：4跳 ε-deploy / 低置信回退 / 错题解剖 / 定稿 spec。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase23"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase4._phase4_common import timed_run, utc_now  # noqa: E402
from phase23._phase23_common import (  # noqa: E402
    CAP, EPS_TIMING_FLOOR, FINE_GRID, FIXED_3_ACC, MIN_N, SEED, TIMING_FLOOR,
    eval_floor_m2, eval_structure_d, filter_by_hop, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_json, load_m2_head_state, load_rich_head, load_splits,
    stats, timing_metrics, write_phase23_result,
)

PHASE24_OUT = ROOT / "results" / "phase24"


def write_phase24_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(24, eid, payload)
