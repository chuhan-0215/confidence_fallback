"""Phase 23 · 收官验证：稳健性 / 分歧融合 / 4跳专项 / 双探针。"""
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

PHASE23_OUT = ROOT / "results" / "phase23"
DATASET = ROOT / "data" / "prosqa_test_graph_4_coconut.json"
M2_HEAD = ROOT / "results" / "phase10" / "m2_enough_stop_head.pt"
CAP = 8
SEED = 99
MIN_N = 3
FIXED_3_ACC = 0.863
TIMING_FLOOR = 0.5
EPS_TIMING_FLOOR = 0.5
FINE_GRID = [round(x * 0.05, 2) for x in range(3, 17)]
ROBUST_SEEDS = (0, 1, 2, 42, 99)


def write_phase23_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(23, eid, payload)


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


def timing_metrics(stop_ns: list, fcs: list, epsilons: tuple = (0, 1, 2)) -> dict:
    valid = [(s, f) for s, f in zip(stop_ns, fcs) if f is not None]
    if not valid:
        return {}
    out = {}
    for eps in epsilons:
        hits = sum(1 for s, f in valid if abs(s - f) <= eps)
        out[f"timing_eps{eps}"] = round(hits / len(valid), 4)
    out["timing_strict"] = out.get("timing_eps0")
    return out


def filter_by_hop(samples, hop: int):
    from boundary_budget import blind_depth
    return [s for s in samples if blind_depth(s) == hop]


def stats(accs: list) -> dict:
    if not accs:
        return {}
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    return {
        "mean": round(mean, 4),
        "stdev": round(var ** 0.5, 4),
        "min": round(min(accs), 4),
        "max": round(max(accs), 4),
        "n_seeds": len(accs),
    }


@torch.no_grad()
def eval_structure_d(model, tokenizer, samples, device, seed, profile):
    from boundary_budget import make_structure_budget_fn
    from evaluate_coconut import expected_answer
    from run_adaptive_stop_experiment import predict_at_n
    from stop_head import first_correct_step

    budget_fn = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    correct = total = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        n = budget_fn(sample)
        pred = predict_at_n(model, tokenizer, sample, n, device, seed=seed + idx * 31, eval_profile=profile)
        expected = expected_answer(sample, profile)
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        total += 1
        if pred == expected:
            correct += 1
        stop_ns.append(n)
        fcs.append(fc)
    row = {"accuracy": round(correct / total, 4), "total": total, "mean_stop_n": round(sum(stop_ns) / total, 2)}
    row.update(timing_metrics(stop_ns, fcs))
    row["params"] = {"uses_oracle": False, "mode": "structure_d"}
    return row


@torch.no_grad()
def eval_floor_m2(head, model, tokenizer, samples, device, seed, profile, thr, floor_fn, mode: str):
    from evaluate_coconut import expected_answer
    from graph_utils import build_eval_prompt
    from run_adaptive_stop_experiment import predict_at_n
    from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step

    correct = total = 0
    stop_ns, fcs = [], []
    pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        floor_n = max(MIN_N, min(CAP, floor_fn(sample)))
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        stop_n = CAP
        final = preds.get(CAP, "")
        prev, streak = "", 0
        for n in range(1, CAP + 1):
            final = preds[n]
            ab, streak, ch = _rich_step_features(final, prev, streak)
            prev = final
            prompt = build_eval_prompt(sample, n, seed=seed + idx * 31 + n,
                choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0), torch.tensor([n], device=device),
                torch.tensor([ab], device=device), torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            stop_n = n
            if n >= floor_n and prob >= thr:
                break
        total += 1
        if final == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {"accuracy": round(correct / total, 4), "total": total, "mean_stop_n": round(sum(stop_ns) / total, 2)}
    row.update(timing_metrics(stop_ns, fcs))
    row["params"] = {"uses_oracle": False, "mode": mode, "threshold": thr}
    return row
