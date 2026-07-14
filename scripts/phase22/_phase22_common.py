"""Phase 22 · 部署定稿：structure_d 二段式 / 跳数分治 / ε-stop / ε-Pareto。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "phase6"))

from phase4._phase4_common import timed_run, utc_now  # noqa: E402
from stop_head import RichStopHead, split_dataset  # noqa: E402

PHASE22_OUT = ROOT / "results" / "phase22"
DATASET = ROOT / "data" / "prosqa_test_graph_4_coconut.json"
M2_HEAD = ROOT / "results" / "phase10" / "m2_enough_stop_head.pt"
CAP = 8
SEED = 99
MIN_N = 3
FIXED_3_ACC = 0.863
TIMING_FLOOR = 0.5
EPS_TIMING_FLOOR = 0.5
FINE_GRID = [round(x * 0.05, 2) for x in range(3, 17)]


def write_phase22_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(22, eid, payload)


def load_json(*rels: str) -> dict:
    for rel in rels:
        for base in (ROOT / "outbox/results/from_a800", ROOT / "results"):
            p = base / rel
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
    return {}


def load_splits(train_ratio: float = 0.6, seed: int = 42):
    from evaluate_coconut import load_dataset
    return split_dataset(load_dataset(DATASET, None), train_ratio=train_ratio, seed=seed)


def load_full_dataset():
    from evaluate_coconut import load_dataset
    return load_dataset(DATASET, None)


def load_m2_head_state(device) -> Optional[dict]:
    if not M2_HEAD.is_file():
        return None
    ckpt = torch.load(M2_HEAD, map_location=device, weights_only=False)
    return ckpt.get("state_dict")


def load_rich_head(device, state_dict: Optional[dict] = None) -> RichStopHead:
    head = RichStopHead(hidden_dim=768, max_steps=CAP, dropout=0.15).to(device)
    if state_dict:
        head.load_state_dict(state_dict)
    return head


def is_feasible(row: dict) -> bool:
    return row["accuracy"] >= FIXED_3_ACC and (row.get("stop_timing_acc") or 0) >= TIMING_FLOOR


def is_eps_deployable(row: dict) -> bool:
    return (
        row["accuracy"] >= FIXED_3_ACC
        and (row.get("timing_eps1") or 0) >= EPS_TIMING_FLOOR
        and row.get("params", {}).get("uses_oracle", False) is not True
    )


def is_deployable_mvp(row: dict) -> bool:
    return (
        row["accuracy"] >= FIXED_3_ACC
        and (row.get("mean_stop_n") or CAP) <= 4.5
        and row.get("params", {}).get("uses_oracle", False) is not True
    )


def epsilon_timing(stop_n: int, fc: int, eps: int) -> bool:
    return abs(stop_n - fc) <= eps


def timing_metrics(stop_ns: list, fcs: list, epsilons: tuple = (0, 1, 2)) -> dict:
    valid = [(s, f) for s, f in zip(stop_ns, fcs) if f is not None]
    if not valid:
        return {}
    out = {}
    for eps in epsilons:
        hits = sum(1 for s, f in valid if epsilon_timing(s, f, eps))
        out[f"timing_eps{eps}"] = round(hits / len(valid), 4)
    out["timing_strict"] = out.get("timing_eps0")
    return out


def epsilon_stop_target(fc: Optional[int], *, min_n: int, cap: int, eps: int = 1) -> int:
    if fc is None:
        return cap
    candidates = [n for n in range(min_n, cap + 1) if abs(n - fc) <= eps]
    if candidates:
        return min(candidates)
    return max(min_n, min(cap, fc))


def row_summary(row: dict, strategy: str, **extra) -> dict:
    eps = {k: row.get(k) for k in ("timing_eps0", "timing_eps1", "timing_eps2") if row.get(k) is not None}
    out = {
        "strategy": strategy,
        "accuracy": row.get("accuracy"),
        "stop_timing_acc": row.get("stop_timing_acc"),
        "mean_stop_n": row.get("mean_stop_n"),
        "feasible": is_feasible(row) if row.get("stop_timing_acc") is not None else False,
        "deployable_mvp": is_deployable_mvp(row),
        "eps_deployable": is_eps_deployable({**row, **eps}) if eps.get("timing_eps1") is not None else None,
        "params": row.get("params") or {},
        **eps,
        **extra,
    }
    return out
