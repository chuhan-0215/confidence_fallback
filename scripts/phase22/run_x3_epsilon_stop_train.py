#!/usr/bin/env python3
"""X3 · ε-stop 标签重训：教 head 在 |stop−fc|≤1 处停，而非精确 fc。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase22_common import (
    CAP, FINE_GRID, MIN_N, SEED, epsilon_stop_target, is_deployable_mvp, is_eps_deployable,
    is_feasible, load_full_dataset, load_m2_head_state, load_rich_head, load_splits,
    row_summary, timed_run, timing_metrics, write_phase22_result,
)
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import (
    RichStopExample, _rich_step_features, calibrate_rich_threshold, encode_answer_bucket,
    evaluate_rich_stop, extract_latent_hidden, first_correct_step, split_train_val_samples,
    train_rich_stop_head,
)


def build_epsilon_stop_examples(
    model, tokenizer, samples, *, cap, min_n, eps, device, seed, predict_fn, expected_fn, eval_profile,
):
    rows = []
    for idx, sample in enumerate(samples):
        fc, preds = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=predict_fn, expected_fn=expected_fn, eval_profile=eval_profile,
        )
        stop_target = epsilon_stop_target(fc, min_n=min_n, cap=cap, eps=eps)
        prev, streak = "", 0
        for n in range(1, cap + 1):
            pred = preds[n]
            ab, streak, ch = _rich_step_features(pred, prev, streak)
            prev = pred
            prompt = build_eval_prompt(sample, n, seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1)
            rows.append(RichStopExample(
                hidden=hid.cpu(), step=n, answer_bucket=ab, streak=streak, changed=ch,
                should_stop=1.0 if n == stop_target else 0.0, sample_idx=idx,
            ))
    return rows


@torch.no_grad()
def collect_stop_eps(head, model, tokenizer, samples, *, cap, threshold, min_n, device, seed,
                     predict_fn, expected_fn, eval_profile):
    from stop_head import evaluate_rich_stop
    row = evaluate_rich_stop(
        head, model, tokenizer, samples, cap=cap, min_n=min_n, threshold=threshold,
        device=device, seed=seed, predict_fn=predict_fn, expected_fn=expected_fn,
        build_prompt_fn=build_eval_prompt, eval_profile=eval_profile,
    )
    stop_ns, fcs = [], []
    head.eval()
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
            prompt = build_eval_prompt(sample, n, seed=seed + idx * 31 + n,
                choice_order=eval_profile.choice_order,
                shuffle_edges=eval_profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n - 1).to(device)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0), torch.tensor([n], device=device),
                torch.tensor([ab], device=device), torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            stop_n = n
            if n >= min_n and prob >= threshold:
                break
        stop_ns.append(stop_n)
    row.update(timing_metrics(stop_ns, fcs))
    row["params"]["uses_oracle"] = False
    row["params"]["label_mode"] = "epsilon_stop"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eps", type=int, default=1)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)

        train_ex = build_epsilon_stop_examples(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, eps=args.eps,
            device=device, seed=42, predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        val_ex = build_epsilon_stop_examples(
            model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, eps=args.eps,
            device=device, seed=43, predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        head, train_metrics = train_rich_stop_head(
            train_ex, val_ex, epochs=args.epochs, device=device, init_state=load_m2_head_state(device),
        )
        ckpt = ROOT / "results" / "phase22" / "x3_epsilon_stop_head.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "train_metrics": train_metrics, "eps": args.eps}, ckpt)

        thr, _ = calibrate_rich_threshold(
            head, model, tokenizer, val_sub, cap=CAP, min_n=MIN_N,
            thresholds=FINE_GRID, device=device, seed=SEED, predict_fn=pfn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
            optimize="timing", min_accuracy=0.863,
        )

        results = []
        for split_name, samples in (("test", test_set), ("full_419", full)):
            row = collect_stop_eps(
                head, model, tokenizer, samples, cap=CAP, threshold=thr, min_n=MIN_N,
                device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
            )
            results.append(row_summary(row, "epsilon_stop_head", split=split_name, threshold=thr, eps=args.eps))

        full_row = next(r for r in results if r["split"] == "full_419")
        return {
            "results": results,
            "train_metrics": train_metrics,
            "threshold": thr,
            "eps": args.eps,
            "full_419": full_row,
            "feasible": full_row.get("feasible"),
            "deployable_mvp": full_row.get("deployable_mvp"),
            "eps_deployable": full_row.get("eps_deployable"),
            "insight": f"ε-stop 标签（ε={args.eps}）：直接优化「差 {args.eps} 步算对」的停步目标。",
            "mentor_brief": (
                f"X3 ε-stop：acc {full_row['accuracy']:.1%} strict {full_row.get('stop_timing_acc', 0):.1%} "
                f"ε=1 {full_row.get('timing_eps1', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "x3_epsilon_stop_train", "X3 · ε-stop 重训", device=args.device)
    write_phase22_result("x3_epsilon_stop_train", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
