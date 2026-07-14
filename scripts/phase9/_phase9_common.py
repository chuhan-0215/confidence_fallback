"""Shared helpers for Phase 9 final closure experiments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))

from phase8._phase8_common import (  # noqa: E402
    DEPLOY_DEFAULTS,
    evaluate_hybrid,
    load_full_dataset,
    load_model_bundle,
    load_test_split,
    timed_run,
    utc_now,
)

PHASE9_OUT = ROOT / "results" / "phase9"


def write_phase9_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(9, experiment_id, payload)
