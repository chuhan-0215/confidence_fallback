"""Phase 30 · 死区专补 + 组合最优（P28 后续）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase23"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase4._phase4_common import timed_run, utc_now  # noqa: E402
from phase23._phase23_common import CAP, MIN_N, SEED, load_json, stats  # noqa: E402

PHASE30_OUT = ROOT / "results" / "phase30"
GAP_INDICES = (111, 189, 261)
ROBUST_SEEDS = (0, 1, 2, 42, 99)


def write_phase30_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(30, eid, payload)
