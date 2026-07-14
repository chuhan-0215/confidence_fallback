"""Phase 33 · 通解跨集修复与统一策略（共享）。"""
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

PHASE33_OUT = ROOT / "results" / "phase33"
TRANSFER_THR = 0.48
TAU_GRID = [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60]
HURT_SLICE_IDS = (
    "v_diamond_5", "push_ext6_from4", "diameter_wide", "prosqa_diameter_wide",
    "push_ext5_from3", "mix_75_4", "mix_50_4", "syn_chain_5_wide", "hops_3",
)
SURGICAL_LOW = 0.48
SURGICAL_HIGH = 0.551


def write_phase33_result(eid: str, payload: dict) -> Path:
    from shared.phase_io import write_phase_result
    return write_phase_result(33, eid, payload)


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
def eval_surgical(head, model, tokenizer, samples, *, device, seed, profile,
                  struct_floor, knn_floor, knn_thr, pfn, low_thr=SURGICAL_LOW, high_thr=SURGICAL_HIGH):
    from _fallback_eval import run_knn_path
    from boundary_budget import blind_depth
    from evaluate_coconut import expected_answer
    from graph_utils import build_eval_prompt
    from run_adaptive_stop_experiment import predict_at_n
    from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step

    head.eval()
    correct = hop3_dual = hop4_zone = hop4_arb = champ = fallback_count = 0
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        hop = blind_depth(sample)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        ps = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        ab0, st0, ch0 = _rich_step_features(ps, "", 1)
        prob_s = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        pk, nk, _, prob_k = run_knn_path(
            head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
            profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        in_zone = low_thr <= prob_s < high_thr
        used_fallback = False
        if hop < 4:
            if ps != pk:
                hop3_dual += 1
                final = pk
            else:
                final = ps
        elif in_zone and ps != pk:
            hop4_zone += 1
            hop4_arb += 1
            used_fallback = True
            final = pk if prob_k >= prob_s else ps
        elif prob_s < low_thr:
            hop4_zone += 1
            used_fallback = True
            final = pk
        else:
            champ += 1
            final = ps
        if used_fallback or (hop < 4 and ps != pk):
            fallback_count += 1
        if final == expected:
            correct += 1
    total = len(samples)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total": total,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "hop3_dual_count": hop3_dual,
        "hop4_zone_count": hop4_zone,
        "hop4_arbitrate_count": hop4_arb,
        "champion_path_count": champ,
        "params": {"low_thr": low_thr, "high_thr": high_thr, "mode": "surgical_deadzone"},
    }


@torch.no_grad()
def eval_dual_threshold(head, model, tokenizer, samples, *, device, seed, profile,
                        struct_floor, knn_floor, knn_thr, pfn,
                        low_thr=TRANSFER_THR, high_thr=0.55):
    from _fallback_eval import run_knn_path
    from evaluate_coconut import expected_answer
    from graph_utils import build_eval_prompt
    from run_adaptive_stop_experiment import predict_at_n
    from stop_head import _rich_step_features, extract_latent_hidden

    head.eval()
    correct = fallback_count = arbitrate_count = 0
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
        prob0 = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        pk, _, _, pk_prob = run_knn_path(
            head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
            profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        if prob0 >= high_thr:
            final = pred0
        elif prob0 < low_thr:
            fallback_count += 1
            final = pk
        elif pred0 != pk:
            arbitrate_count += 1
            fallback_count += 1
            final = pk if pk_prob >= prob0 else pred0
        else:
            final = pred0
        if final == expected:
            correct += 1
    total = len(samples)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total": total,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "arbitrate_count": arbitrate_count,
        "params": {"low_thr": low_thr, "high_thr": high_thr, "mode": "dual_threshold"},
    }
