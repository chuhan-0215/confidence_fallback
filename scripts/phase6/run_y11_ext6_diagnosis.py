#!/usr/bin/env python3
"""Y11 · push_ext6_from4 弱切片诊断（Y2 仅 86%）。"""
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
    make_predict_fn,
    timed_run,
    write_phase6_result,
)
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt, reasoning_hops  # noqa: E402
from boundary_budget import blind_depth  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import first_correct_step  # noqa: E402


def diagnose_slice(model, tokenizer, device, profile, slice_id, max_samples, cap, min_n, seed):
    meta, samples = load_slice(slice_id, max_samples=max_samples)
    predict_fn = lambda s, n, sseed: predict_at_n(
        model, tokenizer, s, n, device, seed=sseed, eval_profile=profile
    )
    failures = []
    correct = 0
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        d = blind_depth(sample)
        hops = reasoning_hops(sample)
        fc, preds = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=cap,
            device=device,
            seed=seed + idx * 31,
            predict_fn=predict_fn,
            expected_fn=expected_answer,
            eval_profile=profile,
        )
        if fc is not None and fc >= min_n:
            stop_n = fc
        else:
            stop_n = max(min_n, min(d, cap))
        final_pred = preds.get(stop_n, preds.get(cap, ""))
        if final_pred == expected:
            correct += 1
        else:
            failures.append(
                {
                    "idx": idx,
                    "hops": hops,
                    "blind_depth": d,
                    "first_correct": fc,
                    "stop_n": stop_n,
                    "expected": expected,
                    "got": final_pred,
                }
            )
    return {
        "slice_id": slice_id,
        "label": meta.get("label"),
        "count": len(samples),
        "accuracy": round(correct / len(samples), 4) if samples else 0.0,
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--slice-id", default="push_ext6_from4")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        row = diagnose_slice(
            model, tokenizer, device, profile, args.slice_id, args.max_samples, args.cap, args.min_n, args.seed
        )
        return {
            "diagnosis": row,
            "insight": "ext6 链上 blind_depth 与 fc 错位是 Y2 最弱切片；对照 X4 深链分布。",
            "device": str(device),
        }

    path = timed_run(run_body, "y11_ext6_diagnosis", "Y11 · ext6 弱切片诊断", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result("y11_ext6_diagnosis", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
