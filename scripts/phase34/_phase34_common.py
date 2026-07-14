"""Phase 34 · Collateral Guard（共享 eval + 常量）。"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase23"))
sys.path.insert(0, str(ROOT / "model"))

from phase23._phase23_common import CAP, MIN_N, M2_HEAD, SEED  # noqa: E402
from shared.eval_paths import eval_main_path, make_slice_row, rollup_slice_rows  # noqa: E402

PHASE34_OUT = ROOT / "results" / "phase34"
TRANSFER_THR = 0.48
T_LOW_GRID = (0.32, 0.35, 0.38, 0.40)
T_MID_GRID = (0.44, 0.45, 0.46, 0.47, 0.48)
HURT_SLICE_IDS = (
    "v_diamond_5", "push_ext6_from4", "diameter_wide", "prosqa_diameter_wide",
    "push_ext5_from3", "mix_75_4", "mix_50_4", "syn_chain_5_wide", "hops_3",
)
CHAMPION_SEEDS = (42, 43, 44)


def write_phase34_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(34, eid, payload)


def m2_head_ready() -> bool:
    return M2_HEAD.is_file()


def unique_slice_ids() -> list[str]:
    from dataset_registry import all_slice_specs
    return [s["id"] for s in all_slice_specs()]


def split_val_test(samples: list, *, val_ratio: float = 0.2, seed: int = 43) -> tuple[list, list]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * val_ratio)) if len(samples) >= 5 else 1
    val_idx = set(idx[:n_val])
    val = [samples[i] for i in range(len(samples)) if i in val_idx]
    test = [samples[i] for i in range(len(samples)) if i not in val_idx]
    if not test:
        test, val = val, test
    return val, test


@torch.no_grad()
def _main_step(head, model, tokenizer, sample, *, device, seed, profile, struct_floor, idx: int):
    from evaluate_coconut import expected_answer
    from graph_utils import build_eval_prompt
    from run_adaptive_stop_experiment import predict_at_n
    from stop_head import _rich_step_features, extract_latent_hidden

    n0 = max(MIN_N, min(CAP, struct_floor(sample)))
    pred0 = predict_at_n(
        model, tokenizer, sample, n0, device,
        seed=seed + idx * 31, eval_profile=profile,
    )
    prompt0 = build_eval_prompt(
        sample, n0, seed=seed + idx * 31,
        choice_order=profile.choice_order,
        shuffle_edges=profile.prompt_mode != "fixed_edges",
    )
    ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
    hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
    ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
    prob0 = torch.sigmoid(head(
        hid0.unsqueeze(0), torch.tensor([n0], device=device),
        torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
        torch.tensor([ch0], device=device),
    )).item()
    return n0, pred0, prob0, expected_answer(sample, profile)


@torch.no_grad()
def _knn_preview(model, tokenizer, sample, *, device, seed, profile, knn_floor, idx: int) -> str:
    from run_adaptive_stop_experiment import predict_at_n

    n_prev = max(MIN_N, min(CAP, knn_floor(sample)))
    return predict_at_n(
        model, tokenizer, sample, n_prev, device,
        seed=seed + idx * 31, eval_profile=profile,
    )


@torch.no_grad()
def _full_knn(head, model, tokenizer, sample, *, device, seed, profile, knn_floor, knn_thr, pfn, idx: int):
    from _fallback_eval import run_knn_path

    pk, nk, _, pk_prob = run_knn_path(
        head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
        profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
    )
    return pk, nk, pk_prob


def _policy_row(mode: str, correct: int, total: int, fallback_count: int, extra: dict | None = None) -> dict:
    row = {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total": total,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "params": {"mode": mode},
    }
    if extra:
        row["params"].update(extra)
    return row


@torch.no_grad()
def eval_agreement_lock(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn, fallback_thr: float = TRANSFER_THR,
    hop4_only: bool = False,
):
    from boundary_budget import blind_depth

    head.eval()
    correct = fallback_count = agreement_skip = 0
    for idx, sample in enumerate(samples):
        _, pred0, prob0, expected = _main_step(
            head, model, tokenizer, sample, device=device, seed=seed,
            profile=profile, struct_floor=struct_floor, idx=idx,
        )
        pred_prev = _knn_preview(
            model, tokenizer, sample, device=device, seed=seed,
            profile=profile, knn_floor=knn_floor, idx=idx,
        )
        if pred0 == pred_prev:
            agreement_skip += 1
            final = pred0
        elif prob0 < fallback_thr and (not hop4_only or blind_depth(sample) >= 4):
            fallback_count += 1
            pk, _, _ = _full_knn(
                head, model, tokenizer, sample, device=device, seed=seed,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn, idx=idx,
            )
            final = pk
        else:
            final = pred0
        if final == expected:
            correct += 1
    return _policy_row(
        "agreement_lock", correct, len(samples), fallback_count,
        {"fallback_thr": fallback_thr, "hop4_only": hop4_only, "agreement_skip": agreement_skip},
    )


@torch.no_grad()
def eval_tri_zone(
    head, model, tokenizer, samples, *, device, seed, profile,
    struct_floor, knn_floor, knn_thr, pfn,
    t_low: float, t_mid: float, hop4_only: bool = False,
):
    from boundary_budget import blind_depth

    head.eval()
    correct = fallback_count = zone_mid = zone_low = 0
    for idx, sample in enumerate(samples):
        _, pred0, prob0, expected = _main_step(
            head, model, tokenizer, sample, device=device, seed=seed,
            profile=profile, struct_floor=struct_floor, idx=idx,
        )
        pred_prev = _knn_preview(
            model, tokenizer, sample, device=device, seed=seed,
            profile=profile, knn_floor=knn_floor, idx=idx,
        )
        do_fallback = False
        if prob0 >= t_mid:
            final = pred0
        elif prob0 < t_low:
            do_fallback = True
            zone_low += 1
        elif pred0 != pred_prev:
            do_fallback = True
            zone_mid += 1
        else:
            final = pred0

        if do_fallback:
            if hop4_only and blind_depth(sample) < 4:
                final = pred0
            else:
                fallback_count += 1
                pk, _, _ = _full_knn(
                    head, model, tokenizer, sample, device=device, seed=seed,
                    profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn, idx=idx,
                )
                final = pk
        if final == expected:
            correct += 1
    return _policy_row(
        "tri_zone", correct, len(samples), fallback_count,
        {"t_low": t_low, "t_mid": t_mid, "hop4_only": hop4_only,
         "zone_low_count": zone_low, "zone_mid_disagree_count": zone_mid},
    )


eval_hop4_tri_zone = eval_tri_zone  # hop4_only=True 调用
