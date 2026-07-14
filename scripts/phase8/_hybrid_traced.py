"""Hybrid eval with per-sample traces for Phase 8 failure/timing analysis."""
from __future__ import annotations

from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from phase7._hybrid_eval import first_probe_n, resolve_stop_n
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


def evaluate_hybrid_traced(
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
) -> dict:
    correct = total = probe_sum = one_probe_hits = timing_hits = timing_total = 0
    failures = []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = min(blind_depth(sample), cap)
        sseed = seed + idx * 31
        n0 = first_probe_n(first_mode, sample, cap)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=sseed, eval_profile=profile)
        probes = 1
        path = "one_shot"
        fc = None
        stop_n = n0

        if pred0 == expected:
            final_pred = pred0
            one_probe_hits += 1
            fc, _preds_scan = first_correct_step(
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
            if fc is not None:
                timing_total += 1
                if fc == n0:
                    timing_hits += 1
        else:
            path = "retry_fc"
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
            charged = {n0}
            for n in range(max(min_n, 1), stop_n + 1):
                charged.add(n)
            probes = len(charged)
            if fc is not None:
                timing_total += 1
                if stop_n == fc:
                    timing_hits += 1

        total += 1
        ok = final_pred == expected
        if ok:
            correct += 1
        else:
            gap = (fc - d) if fc is not None else None
            cat = "never_first_correct" if fc is None else (
                "early_fc_deep_misbudget" if fc is not None and fc < d and gap is not None and gap <= -2 else
                "retry_exhausted" if path == "retry_fc" else "one_shot_miss"
            )
            failures.append(
                {
                    "idx": idx,
                    "path": path,
                    "category": cat,
                    "hops": blind_depth(sample),
                    "blind_depth": d,
                    "first_correct": fc,
                    "stop_n": stop_n,
                    "first_probe_n": n0,
                    "probes": probes,
                    "gap_fc_minus_d": gap,
                    "expected": expected,
                    "got": final_pred,
                }
            )
        probe_sum += probes

    return {
        "strategy": f"hybrid_{first_mode}_then_{retry_mode}",
        "accuracy": round(correct / total, 4) if total else 0.0,
        "mean_forward_probes": round(probe_sum / total, 3) if total else 0.0,
        "one_probe_success_rate": round(one_probe_hits / total, 4) if total else 0.0,
        "stop_timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
        "correct": correct,
        "total": total,
        "wrong_count": len(failures),
        "failures": failures,
        "by_category": _count_key(failures, "category"),
        "by_path": _count_key(failures, "path"),
    }


def _count_key(rows: list, key: str) -> dict:
    out: dict = {}
    for row in rows:
        k = row.get(key) or "unknown"
        out[k] = out.get(k, 0) + 1
    return out
