"""Shared helpers for Phase 8 production-closure experiments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))
sys.path.insert(0, str(ROOT / "scripts" / "phase7"))

from phase6._phase6_common import load_model_bundle, load_test_split, timed_run, utc_now  # noqa: E402
from phase7._hybrid_eval import evaluate_hybrid  # noqa: E402

PHASE8_OUT = ROOT / "results" / "phase8"
DEPLOY_DEFAULTS = {
    "strategy": "two_probe_n_eq_d_then_soft_floor",
    "first_mode": "n_eq_d",
    "retry_mode": "soft_floor",
    "min_n": 2,
    "cap": 8,
    "seed": 99,
}


def write_phase8_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(8, experiment_id, payload)


def load_full_dataset(max_samples=None):
    from evaluate_coconut import load_dataset

    return load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", max_samples)
