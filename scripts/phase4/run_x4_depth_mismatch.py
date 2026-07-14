#!/usr/bin/env python3
"""X4 · 思考深度错位：first_correct 步数 vs blind_depth 的差值分布（元认知）。"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase4_common import load_model_bundle, timed_run  # noqa: E402
from boundary_budget import blind_depth  # noqa: E402
from evaluate_coconut import expected_answer, load_dataset  # noqa: E402
from graph_utils import reasoning_hops  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import first_correct_step  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    args = ap.parse_args()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    dataset = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", args.max_samples)

    mismatch = Counter()
    by_hop = {"3": Counter(), "4": Counter()}
    never_correct = 0

    for idx, sample in enumerate(dataset):
        d = blind_depth(sample)
        hops = reasoning_hops(sample)
        fc, _ = first_correct_step(
            model,
            tokenizer,
            sample,
            cap=args.cap,
            device=device,
            seed=42 + idx * 31,
            predict_fn=lambda s, n, seed: predict_at_n(
                model, tokenizer, s, n, device, seed=seed, eval_profile=profile
            ),
            expected_fn=expected_answer,
            eval_profile=profile,
        )
        if fc is None:
            never_correct += 1
            mismatch["never"] += 1
            continue
        gap = fc - d
        mismatch[str(gap)] += 1
        hop_key = "3" if hops == 3 else "4"
        by_hop[hop_key][str(gap)] += 1

    path = timed_run(
        lambda: {
            "mismatch_fc_minus_blind_depth": dict(mismatch),
            "by_reasoning_hops": {k: dict(v) for k, v in by_hop.items()},
            "never_correct_count": never_correct,
            "sample_count": len(dataset),
            "insight": "gap=0 表示结构深度刚好；gap>0 需多想；gap<0 结构高估",
        },
        "x4_depth_mismatch",
        "X4 · 思考深度错位分布",
        device=args.device,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
