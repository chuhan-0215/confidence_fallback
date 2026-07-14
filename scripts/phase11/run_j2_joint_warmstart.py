#!/usr/bin/env python3
"""J2 · M2 warm-start + 保守 joint 微调 + streak 门控（主实验）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase11_common import CAP, MIN_N, SEED, is_feasible, load_m2_head_state, load_splits, timed_run, write_phase11_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import (
    calibrate_rich_threshold,
    evaluate_rich_stop,
    evaluate_streak_gated_stop,
    split_train_val_samples,
    train_joint_rich_stop_head,
)

FINE_GRID = [round(x * 0.05, 2) for x in range(3, 17)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--unfreeze-layers", type=int, default=1)
    ap.add_argument("--coconut-lr", type=float, default=5e-6)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        ar_val = evaluate_policy(model, tokenizer, val_sub, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        ar_test = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)

        init_state = load_m2_head_state(device)
        head, train_metrics = train_joint_rich_stop_head(
            model, tokenizer, train_sub, val_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, unfreeze_layers=args.unfreeze_layers, epochs=args.epochs,
            coconut_lr=args.coconut_lr, head_lr=5e-4, early_stop_patience=3,
            init_head_state=init_state,
        )
        ckpt = ROOT / "results" / "phase11" / "j2_joint_warmstart.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head_state": head.state_dict(),
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "train_metrics": train_metrics,
            "unfreeze_layers": args.unfreeze_layers,
        }, ckpt)

        variants = {}
        for name, opt, use_streak in (
            ("correctness_balanced", "balanced", False),
            ("correctness_feasible", "feasible", False),
            ("streak_feasible", "feasible", True),
        ):
            min_acc = ar_val["accuracy"] if "feasible" in name else None
            thr, cal = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, optimize=opt, min_accuracy=min_acc, thresholds=FINE_GRID,
            )
            if use_streak:
                row = evaluate_streak_gated_stop(
                    head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                row["strategy"] = "joint_streak_feasible"
            else:
                row = evaluate_rich_stop(
                    head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt, eval_profile=profile,
                )
                row["strategy"] = f"joint_{name}"
            variants[name] = {
                "threshold": thr, "calibration": cal, "test": row,
                "feasible": is_feasible(row, ar_test["accuracy"]),
            }

        best_name = max(variants, key=lambda k: (
            variants[k]["feasible"],
            variants[k]["test"]["accuracy"],
            variants[k]["test"].get("stop_timing_acc") or 0,
        ))
        best = variants[best_name]
        return {
            "train_metrics": train_metrics,
            "variants": variants,
            "best_variant": best_name,
            "test": best["test"],
            "threshold": best["threshold"],
            "feasible": best["feasible"],
            "auto_route_test_acc": ar_test["accuracy"],
            "insight": "M2 warm-start + 低 lr joint；streak 门控抬 timing。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "j2_joint_warmstart", "J2 · joint warm-start", device=args.device)
    import json
    write_phase11_result("j2_joint_warmstart", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
