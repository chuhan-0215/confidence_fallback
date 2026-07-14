"""Phase 10 · 导师 MVP：模型学会「想够了就停」。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase4._phase4_common import load_model_bundle, timed_run, utc_now  # noqa: E402
from phase5._phase5_common import load_test_split  # noqa: E402
from stop_head import split_dataset  # noqa: E402

PHASE10_OUT = ROOT / "results" / "phase10"
DATASET = ROOT / "data" / "prosqa_test_graph_4_coconut.json"
CAP = 8
MIN_N = 2
SEED = 99


def write_phase10_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(10, experiment_id, payload)


def load_splits():
    from evaluate_coconut import load_dataset

    dataset = load_dataset(DATASET, None)
    train_set, test_set = split_dataset(dataset, train_ratio=0.6, seed=42)
    return train_set, test_set
