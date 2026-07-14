"""Shared helpers for Phase 7 deploy-lock experiments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase6._phase6_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    timed_run,
    utc_now,
)

PHASE7_OUT = ROOT / "results" / "phase7"


def write_phase7_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(7, experiment_id, payload)
