#!/usr/bin/env python3
"""W1 · ε-timing 审计：严格 timing 之外，±1/±2 步有多宽松？"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase21_common import (
    CAP, FINE_GRID, FIXED_3_ACC, SEED, TIMING_FLOOR, is_deployable_mvp, is_feasible,
    load_full_dataset, load_json, load_m2_head_state, load_rich_head, load_splits,
    timed_run, timing_metrics, write_phase21_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    _rich_step_features, calibrate_rich_threshold, evaluate_rich_stop,
    extract_latent_hidden, first_correct_step, split_train_val_samples,
)


@torch.no_grad()
def collect_stop_and_fc(head, model, tokenizer, samples, *, cap, threshold, min_n, device, seed,
                        predict_fn, expected_fn, build_prompt_fn, eval_profile):
    head.eval()
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        fcs.append(fc)
        stop_n = cap
        prev, streak = "", 0
        for n in range(1, cap + 1):
            pred = preds[n]
            ab, streak, ch = _rich_step_features(pred, prev, streak)
            prev = pred
            prompt = build_prompt_fn(sample, n, seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0),
                torch.tensor([n], device=device),
                torch.tensor([ab], device=device),
                torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            stop_n = n
            if n >= min_n and prob >= threshold:
                break
        stop_ns.append(stop_n)
    return stop_ns, fcs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        full = load_full_dataset()
        head = load_rich_head(device, load_m2_head_state(device))
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        bld = build_eval_prompt

        configs = []
        for min_n in (2, 3):
            thr, _ = calibrate_rich_threshold(
                head, model, tokenizer, val_sub, cap=CAP, min_n=min_n,
                thresholds=FINE_GRID, device=device, seed=SEED, predict_fn=pfn,
                expected_fn=expected_answer, build_prompt_fn=bld, eval_profile=profile,
                optimize="timing", min_accuracy=FIXED_3_ACC,
            )
            for split_name, samples in (("test", test_set), ("full_419", full)):
                row = evaluate_rich_stop(
                    head, model, tokenizer, samples, cap=CAP, threshold=thr,
                    min_n=min_n, device=device, seed=SEED, predict_fn=pfn,
                    expected_fn=expected_answer, build_prompt_fn=bld, eval_profile=profile,
                )
                row["params"]["uses_oracle"] = False
                stop_ns, fcs = collect_stop_and_fc(
                    head, model, tokenizer, samples, cap=CAP, threshold=thr, min_n=min_n,
                    device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
                    build_prompt_fn=bld, eval_profile=profile,
                )
                eps = timing_metrics(stop_ns, fcs)
                configs.append({
                    "min_n": min_n, "threshold": thr, "split": split_name,
                    "accuracy": row["accuracy"], "mean_stop_n": row["mean_stop_n"],
                    "stop_timing_acc": row.get("stop_timing_acc"),
                    **eps,
                    "feasible_strict": is_feasible(row),
                    "feasible_eps1": row["accuracy"] >= FIXED_3_ACC and (eps.get("timing_eps1") or 0) >= TIMING_FLOOR,
                    "feasible_eps2": row["accuracy"] >= FIXED_3_ACC and (eps.get("timing_eps2") or 0) >= TIMING_FLOOR,
                })

        full_min3 = next(c for c in configs if c["split"] == "full_419" and c["min_n"] == 3)
        p20 = load_json("phase20/v5_proof_rollup_latest.json")
        return {
            "configs": configs,
            "phase20_best_timing": (p20.get("best") or {}).get("timing"),
            "insight": (
                "若 ε=1 timing 显著高于严格 37%，说明瓶颈在「精确停步」而非「大致够步」；"
                "可重新定义通用 deploy 指标（ε-timing）。"
            ),
            "mentor_brief": (
                f"ε-timing 审计：严格 timing {full_min3.get('stop_timing_acc', 0):.1%}；"
                f"ε=1 {full_min3.get('timing_eps1', 0):.1%} ε=2 {full_min3.get('timing_eps2', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "w1_epsilon_timing", "W1 · ε-timing 审计", device=args.device)
    write_phase21_result("w1_epsilon_timing", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
