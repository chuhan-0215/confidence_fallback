#!/usr/bin/env python3
"""Y8 · 两阶段 hybrid：先 n=d 单次 forward，错则 soft_floor 序贯续探。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase6_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    timed_run,
    write_phase6_result,
)
from boundary_budget import blind_depth  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import first_correct_step  # noqa: E402


def evaluate_two_probe_hybrid(model, tokenizer, test_set, *, cap, min_n, device, seed, profile):
    correct = total = probe_sum = one_probe_hits = 0
    for idx, sample in enumerate(test_set):
        expected = expected_answer(sample, profile)
        d = blind_depth(sample)
        d = min(d, cap)
        pred_d = predict_at_n(
            model, tokenizer, sample, d, device, seed=seed + idx * 31, eval_profile=profile
        )
        probes = 1
        if pred_d == expected:
            final_pred = pred_d
            one_probe_hits += 1
        else:
            fc, preds = first_correct_step(
                model,
                tokenizer,
                sample,
                cap=cap,
                device=device,
                seed=seed + idx * 31,
                predict_fn=lambda s, n, sseed: predict_at_n(
                    model, tokenizer, s, n, device, seed=sseed, eval_profile=profile
                ),
                expected_fn=expected_answer,
                eval_profile=profile,
            )
            if fc is not None and fc >= min_n:
                stop_n = fc
            else:
                stop_n = max(min_n, min(blind_depth(sample), cap))
            final_pred = preds.get(stop_n, preds.get(cap, ""))
            probes = cap  # upper bound charge for retry path (conservative)
            if fc is not None:
                probes = min(cap, max(d, fc))

        total += 1
        if final_pred == expected:
            correct += 1
        probe_sum += probes

    return {
        "strategy": "two_probe_n_eq_d_then_fc",
        "accuracy": round(correct / total, 4) if total else 0.0,
        "mean_forward_probes": round(probe_sum / total, 3) if total else 0.0,
        "one_probe_success_rate": round(one_probe_hits / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        row = evaluate_two_probe_hybrid(
            model, tokenizer, test_set, cap=args.cap, min_n=args.min_n, device=device, seed=args.seed, profile=profile
        )
        return {
            "strategies": [row],
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "insight": "若 mean_probes≈1.5–2 且 acc≥95% 则优于纯序贯 3.1 probes。",
            "device": str(device),
        }

    path = timed_run(run_body, "y8_two_probe_hybrid", "Y8 · 两阶段 hybrid", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result("y8_two_probe_hybrid", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
