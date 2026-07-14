#!/usr/bin/env python3
"""X1 · 算力–准确率 Pareto：对比多种推理范式（不限于自停）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase4_common import load_model_bundle, timed_run  # noqa: E402
from boundary_budget import evaluate_upfront_budget_stop, make_structure_budget_fn  # noqa: E402
from evaluate_coconut import expected_answer, load_dataset  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from run_auto_submit_experiment import evaluate_policy, make_policies  # noqa: E402
from stop_head_tracks import evaluate_soft_floor_first_correct_stop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    args = ap.parse_args()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    dataset = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", args.max_samples)
    policies = make_policies(cap=args.cap)
    rows = []

    for name in ("fixed_3", "auto_route", "auto_route_zero"):
        r = evaluate_policy(
            model, tokenizer, dataset, policies[name], device, cap=args.cap, eval_profile=profile
        )
        hist = r["n_latent_histogram"]
        total = sum(int(v) for v in hist.values())
        mean_n = sum(int(k) * int(v) for k, v in hist.items()) / total if total else 0
        rows.append(
            {
                "strategy": name,
                "accuracy": r["accuracy"],
                "mean_forward_probes": 1.0,
                "mean_stop_step": round(mean_n, 3),
                "pareto_score": round(r["accuracy"], 4),
            }
        )

    sf = evaluate_soft_floor_first_correct_stop(
        model,
        tokenizer,
        dataset,
        cap=args.cap,
        min_n=2,
        device=device,
        seed=42,
        predict_fn=lambda s, n, seed: predict_at_n(
            model, tokenizer, s, n, device, seed=seed, eval_profile=profile
        ),
        expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt,
        eval_profile=profile,
    )
    mean_fc_probes = sf["mean_stop_n"]
    rows.append(
        {
            "strategy": "soft_floor_fc",
            "accuracy": sf["accuracy"],
            "mean_forward_probes": mean_fc_probes,
            "mean_stop_step": mean_fc_probes,
            "stop_timing_acc": sf.get("stop_timing_acc"),
            "pareto_score": round(sf["accuracy"] / max(mean_fc_probes, 1.0), 4),
        }
    )

    budget_fn = make_structure_budget_fn(min_n=1, cap=args.cap)
    up = evaluate_upfront_budget_stop(
        model,
        tokenizer,
        dataset,
        budget_fn,
        strategy_id="blind_depth_once",
        strategy_label="n=blind_depth · 1×forward",
        cap=args.cap,
        min_n=1,
        device=device,
        seed=99,
        predict_fn=predict_at_n,
        expected_fn=expected_answer,
        eval_profile=profile,
    )
    rows.append(
        {
            "strategy": "blind_depth_once",
            "accuracy": up["accuracy"],
            "mean_forward_probes": 1.0,
            "mean_stop_step": up["mean_stop_n"],
            "stop_timing_acc": up.get("stop_timing_acc"),
            "pareto_score": round(up["accuracy"], 4),
        }
    )

    rows.sort(key=lambda x: (-x["accuracy"], x["mean_forward_probes"]))
    path = timed_run(
        lambda: {"strategies": rows, "sample_count": len(dataset), "cap": args.cap},
        "x1_pareto_efficiency",
        "X1 · 算力–准确率 Pareto 对比",
        device=args.device,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
