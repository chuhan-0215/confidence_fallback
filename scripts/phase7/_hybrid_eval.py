"""Hybrid stop evaluators shared by Phase 7 experiments."""
from __future__ import annotations

from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


def first_probe_n(mode: str, sample: dict, cap: int) -> int:
    d = min(blind_depth(sample), cap)
    if mode == "n_eq_d":
        return d
    if mode == "n_d_minus1":
        if d >= 4:
            return max(2, min(d - 1, cap))
        return d
    if mode == "n_eq_3":
        return min(3, d)
    raise ValueError(f"unknown first_probe mode: {mode}")


def resolve_stop_n(
    *,
    mode: str,
    fc: int | None,
    d: int,
    min_n: int,
    cap: int,
    split_depth: int = 4,
    guard_depth: int = 4,
) -> int:
    floor_n = max(min_n, min(d, cap))
    effective_fc = fc
    if mode == "early_fc_guard" and fc == 1 and d >= guard_depth:
        effective_fc = None

    if mode in ("soft_floor", "early_fc_guard"):
        if effective_fc is not None and effective_fc >= min_n:
            return effective_fc
        return floor_n

    if mode == "hop_split":
        if d >= split_depth:
            if effective_fc is not None and effective_fc >= min_n:
                return effective_fc
            return floor_n
        if effective_fc is not None and effective_fc >= floor_n:
            return effective_fc
        return floor_n

    raise ValueError(f"unknown retry mode: {mode}")


def evaluate_hybrid(
    model,
    tokenizer,
    samples,
    *,
    first_mode: str = "n_eq_d",
    retry_mode: str = "soft_floor",
    cap: int,
    min_n: int,
    device,
    seed: int,
    profile,
    split_depth: int = 4,
    guard_depth: int = 4,
    count_probes_exact: bool = False,
) -> dict:
    correct = total = probe_sum = one_probe_hits = 0
    strategy = f"hybrid_{first_mode}_then_{retry_mode}"
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = min(blind_depth(sample), cap)
        sseed = seed + idx * 31
        n0 = first_probe_n(first_mode, sample, cap)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=sseed, eval_profile=profile)
        probes = 1

        if pred0 == expected:
            final_pred = pred0
            one_probe_hits += 1
        else:
            fc, preds = first_correct_step(
                model,
                tokenizer,
                sample,
                cap=cap,
                device=device,
                seed=sseed,
                predict_fn=lambda s, n, ss: predict_at_n(
                    model, tokenizer, s, n, device, seed=ss, eval_profile=profile
                ),
                expected_fn=expected_answer,
                eval_profile=profile,
            )
            stop_n = resolve_stop_n(
                mode=retry_mode,
                fc=fc,
                d=d,
                min_n=min_n,
                cap=cap,
                split_depth=split_depth,
                guard_depth=guard_depth,
            )
            final_pred = preds.get(stop_n, preds.get(d, ""))
            if count_probes_exact:
                charged = {n0}
                start = max(min_n, 1)
                for n in range(start, stop_n + 1):
                    charged.add(n)
                probes = len(charged)
            else:
                probes = min(cap, max(n0, stop_n)) if fc is not None else min(cap, max(n0, d))

        total += 1
        if final_pred == expected:
            correct += 1
        probe_sum += probes

    return {
        "strategy": strategy,
        "first_mode": first_mode,
        "retry_mode": retry_mode,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "mean_forward_probes": round(probe_sum / total, 3) if total else 0.0,
        "one_probe_success_rate": round(one_probe_hits / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
    }


# Back-compat alias used by Y12/Y14
def evaluate_two_probe_hybrid(model, tokenizer, test_set, *, cap, min_n, device, seed, profile):
    return evaluate_hybrid(
        model,
        tokenizer,
        test_set,
        first_mode="n_eq_d",
        retry_mode="soft_floor",
        cap=cap,
        min_n=min_n,
        device=device,
        seed=seed,
        profile=profile,
        count_probes_exact=False,
    )
