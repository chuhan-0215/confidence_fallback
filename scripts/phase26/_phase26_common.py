"""Phase 26 · 终稿锁定：跳数分阈值 / 缺口法医 / ICAIS 数字定稿。"""
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
from phase23._phase23_common import (  # noqa: E402
    CAP, MIN_N, SEED, load_full_dataset, load_json, stats,
)

PHASE26_OUT = ROOT / "results" / "phase26"
ROBUST_SEEDS = (0, 1, 2, 42, 99)


def write_phase26_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(26, eid, payload)
