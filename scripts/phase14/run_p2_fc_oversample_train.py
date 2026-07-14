#!/usr/bin/env python3
"""P2 · first_correct + 正样本 3× 过采样（抬 timing 专用）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase14_common import CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_feasible, load_m2_head_state, load_splits, timed_run, write_phase14_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import (
    RichStopExample,
    build_rich_stop_examples_for_samples,
    calibrate_rich_threshold,
    evaluate_rich_stop,
    split_train_val_samples,
    train_rich_stop_head,
)


def _oversample_fc(examples: list[RichStopExample], factor: int = 3) -> list[RichStopExample]:
    out = list(examples)
    for ex in examples:
        if ex.should_stop > 0.5:
            out.extend([ex] * (factor - 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, train_sub, cap=CAP, device=device, seed=42,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="first_correct",
        )
        train_ex = _oversample_fc(train_ex, factor=3)
        val_ex = build_rich_stop_examples_for_samples(
            model, tokenizer, val_sub, cap=CAP, device=device, seed=43,
            predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
            eval_profile=profile, label_mode="first_correct",
        )
        head, train_metrics = train_rich_stop_head(
            train_ex, val_ex, epochs=args.epochs, device=device,
            init_head_state=load_m2_head_state(device),
        )
        ckpt = ROOT / "results" / "phase14" / "p2_fc_oversample_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics}, ckpt)

        variants = {}
        for opt in ("timing", "balanced"):
            thr, cal = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=SEED,
                predict_fn=pfn, expected_fn=expected_answer, build_prompt_fn=build_eval_prompt,
                eval_profile=profile, optimize=opt, min_accuracy=0.863 if opt == "timing" else None,
                thresholds=FINE_GRID,
            )
            row = evaluate_rich_stop(
                head, model, tokenizer, test_set, cap=CAP, min_n=MIN_N, threshold=thr,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            row["params"]["uses_oracle"] = False
            variants[opt] = {"threshold": thr, "calibration": cal, "test": row,
                             "feasible": is_feasible(row), "deployable_mvp": is_deployable_mvp(row)}

        best = max(variants, key=lambda k: (variants[k]["feasible"], variants[k]["test"].get("stop_timing_acc") or 0, variants[k]["test"]["accuracy"]))
        policies = make_policies(cap=CAP)
        auto = evaluate_policy(model, tokenizer, test_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        return {
            "train_metrics": train_metrics,
            "variants": variants,
            "best_variant": best,
            "test": variants[best]["test"],
            "feasible": variants[best]["feasible"],
            "deployable_mvp": variants[best]["deployable_mvp"],
            "auto_route_test_acc": auto["accuracy"],
            "insight": "fc 正样本 3× 过采样 + timing 校准。",
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
        }

    path = timed_run(run_body, "p2_fc_oversample_train", "P2 · fc 过采样", device=args.device)
    import json
    write_phase14_result("p2_fc_oversample_train", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
