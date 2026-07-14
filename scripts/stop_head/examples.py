from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stop_head.latent import extract_latent_hidden
from stop_head.models import RichStopExample, StopExample, encode_answer_bucket

def first_correct_step(
    model,
    tokenizer,
    sample: dict,
    *,
    cap: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    eval_profile,
) -> Tuple[Optional[int], Dict[int, str]]:
    preds_by_step: Dict[int, str] = {}
    expected = expected_fn(sample, eval_profile)
    first_correct = None
    for n in range(1, cap + 1):
        pred = predict_fn(
            model,
            tokenizer,
            sample,
            n,
            device,
            seed=seed + n,
            eval_profile=eval_profile,
        )
        preds_by_step[n] = pred
        if first_correct is None and pred == expected:
            first_correct = n
    return first_correct, preds_by_step


def build_stop_examples_for_samples(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
) -> List[StopExample]:
    rows: List[StopExample] = []
    for idx, sample in enumerate(samples):
        first_correct, _ = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        stop_target = first_correct if first_correct is not None else cap
        for n in range(1, cap + 1):
            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1)
            should_stop = 1.0 if n == stop_target else 0.0
            rows.append(
                StopExample(
                    hidden=hidden.cpu(),
                    step=n,
                    should_stop=should_stop,
                    sample_idx=idx,
                )
            )
        if progress_cb:
            progress_cb(idx + 1, len(samples))
    return rows


def resolve_hybrid_stop_target(
    sample: dict,
    *,
    model,
    tokenizer,
    cap: int,
    min_n: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    eval_profile,
    first_mode: str = "n_eq_d",
    retry_mode: str = "soft_floor",
) -> int:
    """Teacher stop step from deploy hybrid (n=d then soft_floor); training labels only."""
    from boundary_budget import blind_depth
    from phase7._hybrid_eval import first_probe_n, resolve_stop_n

    expected = expected_fn(sample, eval_profile)
    d = min(blind_depth(sample), cap)
    n0 = first_probe_n(first_mode, sample, cap)
    pred0 = predict_fn(sample, n0, seed)
    if pred0 == expected:
        return n0
    fc, _ = first_correct_step(
        model,
        tokenizer,
        sample,
        cap=cap,
        device=device,
        seed=seed,
        predict_fn=predict_fn,
        expected_fn=expected_fn,
        eval_profile=eval_profile,
    )
    return resolve_stop_n(
        mode=retry_mode,
        fc=fc,
        d=d,
        min_n=min_n,
        cap=cap,
    )


def build_rich_stop_examples_for_samples(
    model,
    tokenizer,
    samples: Sequence[dict],
    *,
    cap: int,
    device: torch.device,
    seed: int,
    predict_fn,
    expected_fn,
    build_prompt_fn,
    eval_profile,
    progress_cb=None,
    label_mode: str = "first_correct",
) -> List[RichStopExample]:
    rows: List[RichStopExample] = []
    for idx, sample in enumerate(samples):
        first_correct, preds_by_step = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_fn,
            eval_profile=eval_profile,
        )
        expected = expected_fn(sample, eval_profile)
        if label_mode == "hybrid_stop":
            stop_target = resolve_hybrid_stop_target(
                sample,
                model=model,
                tokenizer=tokenizer,
                cap=cap,
                min_n=2,
                device=device,
                seed=seed + idx * 31,
                predict_fn=predict_fn,
                expected_fn=expected_fn,
                eval_profile=eval_profile,
            )
        else:
            stop_target = first_correct if first_correct is not None else cap
        prev_pred = ""
        streak = 0
        for n in range(1, cap + 1):
            pred = preds_by_step[n]
            if pred and pred == prev_pred:
                streak += 1
            else:
                streak = 1
            changed = 1.0 if prev_pred and pred != prev_pred else 0.0
            prev_pred = pred

            prompt = build_prompt_fn(
                sample,
                n,
                seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges",
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1)
            if label_mode == "is_correct":
                should_stop = 1.0 if pred == expected else 0.0
            elif label_mode == "stable_correct":
                should_stop = 1.0 if pred == expected and streak >= 2 else 0.0
            elif label_mode == "earliest_stop":
                fc_n = first_correct if first_correct is not None else cap
                first_ok = next((k for k in range(1, cap + 1) if preds_by_step[k] == expected), cap)
                stop_target = min(fc_n, first_ok)
                should_stop = 1.0 if n == stop_target else 0.0
            elif label_mode == "hybrid_stop":
                should_stop = 1.0 if n == stop_target else 0.0
            else:
                should_stop = 1.0 if n == stop_target else 0.0
            rows.append(
                RichStopExample(
                    hidden=hidden.cpu(),
                    step=n,
                    answer_bucket=encode_answer_bucket(pred),
                    streak=streak,
                    changed=changed,
                    should_stop=should_stop,
                    sample_idx=idx,
                )
            )
        if progress_cb:
            progress_cb(idx + 1, len(samples))
    return rows
