#!/usr/bin/env python3
"""Y6 · min_n=2 失败样本分类：hop / gap / misbudget / never_fc。"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from graph_utils import build_eval_prompt, reasoning_hops  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import first_correct_step  # noqa: E402


def classify_failure(*, fc, d, stop_n, hops, expected, got):
    if fc is None:
        return "never_first_correct"
    gap = fc - d
    if gap <= -2:
        cat = "early_fc_deep_misbudget"
    elif gap == -1:
        cat = "fc_one_below_blind"
    elif gap == 0:
        cat = "fc_at_blind_depth"
    else:
        cat = "fc_after_blind"
    if stop_n != fc and fc is not None:
        cat += "_stop_not_fc"
    if hops >= 6:
        cat = "long_chain_" + cat
    return cat


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
        predict_fn = lambda s, n, seed: predict_at_n(
            model, tokenizer, s, n, device, seed=seed, eval_profile=profile
        )

        correct = 0
        by_category = Counter()
        by_hop = Counter()
        failures = []

        for idx, sample in enumerate(test_set):
            expected = expected_answer(sample, profile)
            d = blind_depth(sample)
            hops = reasoning_hops(sample)
            fc, preds = first_correct_step(
                model,
                tokenizer,
                sample,
                cap=args.cap,
                device=device,
                seed=args.seed + idx * 31,
                predict_fn=predict_fn,
                expected_fn=expected_answer,
                eval_profile=profile,
            )
            if fc is not None and fc >= args.min_n:
                stop_n = fc
            else:
                stop_n = max(args.min_n, min(d, args.cap))
            final_pred = preds.get(stop_n, preds.get(args.cap, ""))
            ok = final_pred == expected
            if ok:
                correct += 1
                continue
            cat = classify_failure(fc=fc, d=d, stop_n=stop_n, hops=hops, expected=expected, got=final_pred)
            by_category[cat] += 1
            by_hop[str(hops)] += 1
            failures.append(
                {
                    "idx": idx,
                    "category": cat,
                    "hops": hops,
                    "blind_depth": d,
                    "first_correct": fc,
                    "stop_n": stop_n,
                    "gap_fc_minus_d": (fc - d) if fc is not None else None,
                    "expected": expected,
                    "got": final_pred,
                }
            )

        n = len(test_set)
        return {
            "strategy": "soft_floor_fc",
            "seed": args.seed,
            "min_n": args.min_n,
            "sample_count": n,
            "accuracy": round(correct / n, 4),
            "wrong_count": n - correct,
            "by_category": dict(by_category),
            "by_hop": dict(by_hop),
            "failures": failures,
            "insight": "对照 X3/X4：欠预算(gap<0)与 never_fc 是主要失败模式候选。",
            "device": str(device),
        }

    path = timed_run(run_body, "y6_failure_taxonomy", "Y6 · 失败样本分类", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result("y6_failure_taxonomy", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
