#!/usr/bin/env python3
"""M2 · 「想够了就停」：is_correct 监督的 RichStopHead（导师 MVP 核心）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase7"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase10_common import CAP, MIN_N, SEED, load_splits, timed_run, write_phase10_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from phase7._hybrid_eval import evaluate_hybrid
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import (
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    evaluate_rich_stop,
    split_train_val_samples,
    train_rich_stop_head,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=args.cap, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="is_correct",
        )
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=args.cap, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="is_correct",
        )
        head, train_metrics = train_rich_stop_head(train_ex, val_ex, epochs=args.epochs, device=device)
        ckpt = ROOT / "results" / "phase10" / "m2_enough_stop_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics, "label_mode": "is_correct"}, ckpt)

        thr, cal = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=args.cap, min_n=MIN_N, device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, optimize="balanced",
        )
        learned = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=args.cap, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        learned["strategy"] = "learned_enough_stop"
        learned["mean_forward_probes"] = learned.get("mean_stop_n")

        policies = make_policies(cap=args.cap)
        ar = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=args.cap, eval_profile=profile)
        hy = evaluate_hybrid(model, tokenizer, test_set, cap=args.cap, min_n=MIN_N, device=device, seed=SEED, profile=profile)

        pareto = [
            {"strategy": "learned_enough_stop", "accuracy": learned["accuracy"],
             "mean_forward_probes": learned["mean_stop_n"], "stop_timing_acc": learned.get("stop_timing_acc"),
             "kind": "learned", "label": "想够了就停 · is_correct"},
            {"strategy": hy["strategy"], "accuracy": hy["accuracy"], "mean_forward_probes": hy["mean_forward_probes"],
             "kind": "teacher"},
            {"strategy": "auto_route", "accuracy": ar["accuracy"], "mean_forward_probes": 1.0, "kind": "baseline"},
        ]
        return {
            "train_metrics": train_metrics,
            "label_mode": "is_correct",
            "calibration": cal,
            "threshold": thr,
            "test": learned,
            "pareto": pareto,
            "feasible": learned["accuracy"] >= ar["accuracy"] and (learned.get("stop_timing_acc") or 0) >= 0.5,
            "insight": "训练目标：当前步答对就应停（适可而止）；推理只看 latent+head，无扫步找 fc。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "m2_learned_enough_stop", "M2 · 想够了就停", device=args.device)
    import json
    write_phase10_result("m2_learned_enough_stop", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
