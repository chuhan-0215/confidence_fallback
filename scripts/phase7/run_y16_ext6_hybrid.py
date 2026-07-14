#!/usr/bin/env python3
"""Y16 · push_ext6_from4 弱切片：soft_floor vs hybrid vs n_eq_d。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hybrid_eval import evaluate_hybrid, first_probe_n
from _phase7_common import load_model_bundle, timed_run, write_phase7_result
from dataset_registry import load_slice
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase5._phase5_common import make_predict_fn
from run_adaptive_stop_experiment import predict_at_n
from stop_head_tracks import evaluate_soft_floor_first_correct_stop

SLICE_ID = "push_ext6_from4"


def eval_n_eq_d(model, tokenizer, samples, *, cap, device, seed, profile):
    correct = 0
    for idx, sample in enumerate(samples):
        n = first_probe_n("n_eq_d", sample, cap)
        pred = predict_at_n(
            model, tokenizer, sample, n, device, seed=seed + idx * 31, eval_profile=profile
        )
        if pred == expected_answer(sample, profile):
            correct += 1
    total = len(samples)
    return {
        "strategy": "n_eq_d",
        "accuracy": round(correct / total, 4) if total else 0.0,
        "mean_forward_probes": 1.0,
        "correct": correct,
        "total": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--slice-id", default=SLICE_ID)
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        meta, samples = load_slice(args.slice_id, max_samples=args.max_samples)
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        rows = [
            eval_n_eq_d(
                model, tokenizer, samples, cap=args.cap, device=device, seed=args.seed, profile=profile
            ),
            {
                **evaluate_hybrid(
                    model,
                    tokenizer,
                    samples,
                    cap=args.cap,
                    min_n=args.min_n,
                    device=device,
                    seed=args.seed,
                    profile=profile,
                )
            },
        ]
        sf = evaluate_soft_floor_first_correct_stop(
            model,
            tokenizer,
            samples,
            cap=args.cap,
            min_n=args.min_n,
            device=device,
            seed=args.seed,
            predict_fn=predict_fn,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
        )
        rows.append(
            {
                "strategy": "soft_floor_fc",
                "accuracy": sf["accuracy"],
                "mean_forward_probes": sf.get("mean_stop_n"),
                "stop_timing_acc": sf.get("stop_timing_acc"),
            }
        )
        rows.sort(key=lambda r: -r["accuracy"])
        return {
            "slice_id": args.slice_id,
            "label": meta.get("label"),
            "count": len(samples),
            "cap": args.cap,
            "strategies": rows,
            "device": str(device),
            "insight": "ext6 长链 stress；hybrid 是否缓解 86% 弱切片。",
        }

    path = timed_run(run_body, "y16_ext6_hybrid", "Y16 · ext6 hybrid", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("y16_ext6_hybrid", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
