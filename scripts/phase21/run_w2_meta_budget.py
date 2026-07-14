#!/usr/bin/env python3
"""W2 · 元预算单次前向：图特征 / Coconut 前缀 / 结构规则，无停步头。"""
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
    CAP, SEED, is_deployable_mvp, is_feasible, load_full_dataset, load_splits, timed_run, write_phase21_result,
)
from boundary_budget import (
    build_prompt_budget_labels,
    evaluate_upfront_budget_stop,
    make_d4_rich_binary_budget_fn,
    make_d_minus_one_budget_fn,
    make_prompt_d4_budget_fn,
    make_structure_budget_fn,
    train_d4_weighted_binary_mlp,
    train_prompt_d4_binary_mlp,
)
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import split_train_val_samples

MIN_N = 3


def _row_to_summary(row: dict, strategy: str) -> dict:
    row = dict(row)
    row["strategy"] = strategy
    row["params"] = {**(row.get("params") or {}), "uses_oracle": False}
    return {
        "strategy": strategy,
        "accuracy": row["accuracy"],
        "stop_timing_acc": row.get("stop_timing_acc"),
        "mean_stop_n": row.get("mean_stop_n"),
        "feasible": is_feasible(row),
        "deployable_mvp": is_deployable_mvp(row),
        "params": row["params"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda m, t, s, n, d, ss, ep: predict_at_n(m, t, s, n, d, seed=ss, eval_profile=ep)

        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
            expected_fn=expected_answer, eval_profile=profile,
        )
        val_rows = build_prompt_budget_labels(
            model, tokenizer, val_sub, cap=CAP, min_n=MIN_N, device=device, seed=43,
            predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
            expected_fn=expected_answer, eval_profile=profile,
        )

        rich_head, rich_meta = train_d4_weighted_binary_mlp(train_rows, val_rows, feature_key="rich_features")
        ckpt_rich = ROOT / "results" / "phase21" / "w2_rich_budget_mlp.pt"
        ckpt_rich.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": rich_head.state_dict(), "meta": rich_meta}, ckpt_rich)

        prompt_head, prompt_meta = train_prompt_d4_binary_mlp(train_rows, val_rows)
        ckpt_prompt = ROOT / "results" / "phase21" / "w2_prompt_budget_mlp.pt"
        torch.save({"state_dict": prompt_head.state_dict(), "meta": prompt_meta}, ckpt_prompt)

        strategies = [
            ("structure_d", make_structure_budget_fn(min_n=MIN_N, cap=CAP)),
            ("d_minus_one", make_d_minus_one_budget_fn(min_n=MIN_N, cap=CAP)),
            ("rich_mlp", make_d4_rich_binary_budget_fn(rich_head, min_n=MIN_N, cap=CAP, device=device)),
            ("prompt_joint_mlp", make_prompt_d4_budget_fn(
                prompt_head, model, tokenizer, min_n=MIN_N, cap=CAP,
                device=device, eval_profile=profile, seed_base=SEED,
            )),
        ]

        results = []
        for sid, budget_fn in strategies:
            for split_name, samples in (("test", test_set), ("full_419", full)):
                row = evaluate_upfront_budget_stop(
                    model, tokenizer, samples, budget_fn,
                    strategy_id=sid, strategy_label=sid,
                    cap=CAP, min_n=MIN_N, device=device, seed=SEED,
                    predict_fn=pfn,
                    expected_fn=expected_answer, eval_profile=profile,
                    extra_params={"uses_oracle": False, "paradigm": "meta_budget_single_forward"},
                )
                summary = _row_to_summary(row, sid)
                summary["split"] = split_name
                results.append(summary)

        full_rows = [r for r in results if r["split"] == "full_419"]
        best_acc = max(full_rows, key=lambda r: r["accuracy"])
        best_mvp = max(full_rows, key=lambda r: (r["deployable_mvp"], r["accuracy"]))
        return {
            "results": results,
            "rich_meta": rich_meta,
            "prompt_meta": prompt_meta,
            "best_acc": best_acc,
            "best_deployable_mvp": best_mvp,
            "feasible_any": any(r["feasible"] for r in full_rows),
            "deployable_mvp_any": any(r["deployable_mvp"] for r in full_rows),
            "insight": "元预算范式：推理单次 predict_at_n，训练可用 oracle 标签；测「新问题通用步数」能力。",
            "mentor_brief": (
                f"W2 元预算：最优 acc {best_acc['accuracy']:.1%} ({best_acc['strategy']})；"
                f"deployable {best_mvp['strategy']} acc {best_mvp['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "w2_meta_budget", "W2 · 元预算单次前向", device=args.device)
    write_phase21_result("w2_meta_budget", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
