"""Phase 14 · 新路线：min_n 扫描 / fc 加权 / 前缀 floor / 可部署 MVP。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase4._phase4_common import timed_run, utc_now  # noqa: E402
from stop_head import RichStopHead, split_dataset  # noqa: E402

PHASE14_OUT = ROOT / "results" / "phase14"
DATASET = ROOT / "data" / "prosqa_test_graph_4_coconut.json"
M2_HEAD = ROOT / "results" / "phase10" / "m2_enough_stop_head.pt"
CAP = 8
MIN_N = 2
SEED = 99
FIXED_3_ACC = 0.863
TIMING_FLOOR = 0.5
FINE_GRID = [round(x * 0.05, 2) for x in range(3, 17)]


def write_phase14_result(experiment_id: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(14, experiment_id, payload)


def load_splits(train_ratio: float = 0.6, seed: int = 42):
    from evaluate_coconut import load_dataset

    dataset = load_dataset(DATASET, None)
    return split_dataset(dataset, train_ratio=train_ratio, seed=seed)


def load_full_dataset():
    from evaluate_coconut import load_dataset

    return load_dataset(DATASET, None)


def load_m2_head_state(device) -> Optional[dict]:
    if not M2_HEAD.is_file():
        return None
    ckpt = __import__("torch").load(M2_HEAD, map_location=device, weights_only=False)
    return ckpt.get("state_dict")


def load_rich_head(device, state_dict: Optional[dict] = None) -> RichStopHead:
    head = RichStopHead(hidden_dim=768, max_steps=CAP, dropout=0.15).to(device)
    if state_dict:
        head.load_state_dict(state_dict)
    return head


def is_feasible(learned: dict, acc_floor: float = FIXED_3_ACC) -> bool:
    return learned["accuracy"] >= acc_floor and (learned.get("stop_timing_acc") or 0) >= TIMING_FLOOR


def is_deployable_mvp(row: dict) -> bool:
    """导师可部署 MVP：acc≥fixed_3、省步数、推理无 oracle。"""
    return (
        row["accuracy"] >= FIXED_3_ACC
        and (row.get("mean_stop_n") or CAP) <= 4.5
        and row.get("params", {}).get("uses_oracle", False) is not True
    )
