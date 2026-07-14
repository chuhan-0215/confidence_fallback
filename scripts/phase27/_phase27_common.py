"""Phase 27 · 替代范式：不依赖 Stop Head 主路径的探索。"""
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
    CAP, MIN_N, SEED, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_json, timing_metrics,
)

PHASE27_OUT = ROOT / "results" / "phase27"
VOTE_SEEDS = (0, 42, 99)


def write_phase27_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(27, eid, payload)
