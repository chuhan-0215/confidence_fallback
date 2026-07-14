"""Phase 28 · 缺口三题靶向 + 互补路由。"""
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
from phase23._phase23_common import CAP, MIN_N, SEED, load_json  # noqa: E402

PHASE28_OUT = ROOT / "results" / "phase28"
GAP_INDICES = (111, 189, 261)


def write_phase28_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(28, eid, payload)
