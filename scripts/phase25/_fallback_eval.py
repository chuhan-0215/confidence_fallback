#!/usr/bin/env python3
"""Phase 25 共享：低置信回退评估逻辑。"""
from __future__ import annotations

import torch

from boundary_budget import blind_depth, build_prompt_budget_labels, make_d4_knn_budget_fn, make_structure_budget_fn, train_d4_knn_bank
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, extract_latent_hidden,
    first_correct_step, split_train_val_samples,
)

from _phase25_common import CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_eps_deployable, timing_metrics


def make_predict_fn(model, tokenizer, device, profile):
    def predict_fn(m, t, s, n, d, seed, eval_profile):
        return predict_at_n(m, t, s, n, d, seed=seed, eval_profile=eval_profile)

    return predict_fn


def setup_fallback_stack(model, tokenizer, device, profile):
    train_set, _ = __import__("_phase25_common", fromlist=["load_splits"]).load_splits()
    train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
    pfn = make_predict_fn(model, tokenizer, device, profile)
    head = __import__("_phase25_common", fromlist=["load_rich_head"]).load_rich_head(
        device, __import__("_phase25_common", fromlist=["load_m2_head_state"]).load_m2_head_state(device))
    struct_floor = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    train_rows = build_prompt_budget_labels(
        model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
        predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
    )
    knn_floor = make_d4_knn_budget_fn(train_d4_knn_bank(train_rows)[0], k=5, min_n=MIN_N, cap=CAP)
    knn_thr, _ = calibrate_rich_threshold(
        head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, thresholds=FINE_GRID,
        device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt, eval_profile=profile, optimize="accuracy",
    )
    return head, struct_floor, knn_floor, knn_thr, pfn


@torch.no_grad()
def run_knn_path(head, model, tokenizer, sample, *, device, seed, profile, knn_floor, knn_thr, pfn):
    floor_n = max(MIN_N, min(CAP, knn_floor(sample)))
    fc, preds = first_correct_step(
        model, tokenizer, sample, cap=CAP, device=device, seed=seed,
        predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
    )
    stop_n = CAP
    final = preds.get(CAP, "")
    final_prob = 0.0
    prev, streak = "", 0
    for n in range(1, CAP + 1):
        final = preds[n]
        ab, streak, ch = _rich_step_features(final, prev, streak)
        prev = final
        prompt = build_eval_prompt(sample, n, seed=seed + n,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
        prob = torch.sigmoid(head(
            hid.unsqueeze(0), torch.tensor([n], device=device),
            torch.tensor([ab], device=device), torch.tensor([streak], device=device),
            torch.tensor([ch], device=device),
        )).item()
        stop_n = n
        final_prob = prob
        if n >= floor_n and prob >= knn_thr:
            break
    return final, stop_n, fc, final_prob


@torch.no_grad()
def eval_confidence_fallback(head, model, tokenizer, samples, *, device, seed, profile,
                             struct_floor, knn_floor, knn_thr, pfn, fallback_thr: float,
                             hop4_only: bool = False, answer_arbitrate: bool = False):
    head.eval()
    correct = fallback_count = arbitrate_count = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
        prob0 = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        do_fallback = prob0 < fallback_thr
        if hop4_only:
            do_fallback = do_fallback and blind_depth(sample) >= 4
        if do_fallback:
            fallback_count += 1
            pk, nk, _, pk_prob = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )
            if answer_arbitrate and pk != pred0:
                arbitrate_count += 1
                if pk_prob >= prob0:
                    final, stop_n = pk, nk
                else:
                    final, stop_n = pred0, n0
            else:
                final, stop_n = pk, nk
        else:
            final, stop_n = pred0, n0
        if final == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(samples), 4),
        "arbitrate_count": arbitrate_count,
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "params": {
            "uses_oracle": False,
            "fallback_thr": fallback_thr,
            "hop4_only": hop4_only,
            "answer_arbitrate": answer_arbitrate,
            "mode": "confidence_fallback",
        },
    }
    row.update(timing_metrics(stop_ns, fcs))
    row["deployable_mvp"] = is_deployable_mvp(row)
    row["eps_deployable"] = is_eps_deployable(row)
    return row
