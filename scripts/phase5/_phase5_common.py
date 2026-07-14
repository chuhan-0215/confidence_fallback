"""Shared helpers for Phase 5 follow-up experiments."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "model"))

from phase4._phase4_common import load_model_bundle, timed_run, utc_now, write_result  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import split_dataset  # noqa: E402

PHASE5_OUT = ROOT / "results" / "phase5"


def write_phase5_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(5, experiment_id, payload)


def load_test_split(max_samples=None, train_ratio: float = 0.6, seed: int = 42):
    from evaluate_coconut import load_dataset

    dataset = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", max_samples)
    _train, test_set = split_dataset(dataset, train_ratio=train_ratio, seed=seed)
    return test_set


def make_predict_fn(model, tokenizer, device, profile):
    def _predict(sample, n, seed):
        return predict_at_n(model, tokenizer, sample, n, device, seed=seed, eval_profile=profile)

    return _predict
