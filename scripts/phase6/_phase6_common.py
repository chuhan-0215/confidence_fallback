"""Shared helpers for Phase 6 validation experiments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))

from phase5._phase5_common import (  # noqa: E402
    load_test_split,
    make_predict_fn,
    write_phase5_result,
)
from phase4._phase4_common import load_model_bundle, timed_run, utc_now  # noqa: E402

PHASE6_OUT = ROOT / "results" / "phase6"


def write_phase6_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(6, experiment_id, payload)
