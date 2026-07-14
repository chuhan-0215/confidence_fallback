#!/usr/bin/env python3
"""K3 · hybrid teacher 蒸馏：学停步时机，推理无规则。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase7"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import (
    CAP, FINE_GRID, MIN_N, SEED, feasible_baseline_acc, is_feasible,
    load_splits, timed_run, write_phase12_result,
)
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
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        ar_val = evaluate_policy(model, tokenizer, val_sub, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        ar_test = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        min_acc = feasible_baseline_acc(ar_val["accuracy"], ar_test["accuracy"])

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="hybrid_stop",
        )
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="hybrid_stop",
        )
        head, train_metrics = train_rich_stop_head(train_ex, val_ex, epochs=args.epochs, device=device)
        ckpt = ROOT / "results" / "phase12" / "k3_hybrid_distill.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics, "label_mode": "hybrid_stop"}, ckpt)

        thr, cal = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=SEED,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, optimize="feasible", min_accuracy=min_acc, thresholds=FINE_GRID,
        )
        learned = evaluate_rich_stop(
            head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        learned["strategy"] = "hybrid_distill_stop"
        hy = evaluate_hybrid(model, tokenizer, test_set, cap=CAP, min_n=MIN_N, device=device, seed=SEED, profile=profile)

        return {
            "train_metrics": train_metrics,
            "label_mode": "hybrid_stop",
            "calibration": cal,
            "threshold": thr,
            "test": learned,
            "teacher": {"strategy": hy["strategy"], "accuracy": hy["accuracy"], "mean_forward_probes": hy["mean_forward_probes"]},
            "feasible": is_feasible(learned, ar_test["accuracy"]),
            "auto_route_test_acc": ar_test["accuracy"],
            "insight": "训练用 hybrid 停步标签，推理只有 head；蒸馏 teacher 上界。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "k3_hybrid_distill", "K3 · hybrid 蒸馏", device=args.device)
    import json
    write_phase12_result("k3_hybrid_distill", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
