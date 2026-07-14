#!/usr/bin/env python3
"""Y17 · 全量 419 Pareto：hybrid vs auto_route vs soft_floor。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase8_common import DEPLOY_DEFAULTS, evaluate_hybrid, load_full_dataset, load_model_bundle, timed_run, write_phase8_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase5._phase5_common import make_predict_fn
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head_tracks import evaluate_soft_floor_first_correct_stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=DEPLOY_DEFAULTS["cap"])
    ap.add_argument("--min-n", type=int, default=DEPLOY_DEFAULTS["min_n"])
    ap.add_argument("--seed", type=int, default=DEPLOY_DEFAULTS["seed"])
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        dataset = load_full_dataset(args.max_samples)
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        policies = make_policies(cap=args.cap)
        rows = []

        for name in ("auto_route", "fixed_3"):
            r = evaluate_policy(model, tokenizer, dataset, policies[name], device, cap=args.cap, eval_profile=profile)
            rows.append({"strategy": name, "accuracy": r["accuracy"], "mean_forward_probes": 1.0, "n": r["total"]})

        sf = evaluate_soft_floor_first_correct_stop(
            model, tokenizer, dataset, cap=args.cap, min_n=args.min_n,
            device=device, seed=args.seed, predict_fn=predict_fn,
            expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        rows.append({
            "strategy": "soft_floor_fc",
            "accuracy": sf["accuracy"],
            "mean_forward_probes": sf.get("mean_stop_n"),
            "stop_timing_acc": sf.get("stop_timing_acc"),
            "n": sf["total"],
        })

        hy = evaluate_hybrid(
            model, tokenizer, dataset, cap=args.cap, min_n=args.min_n,
            device=device, seed=args.seed, profile=profile,
        )
        rows.append({**hy, "n": hy["total"]})
        rows.sort(key=lambda r: (-r["accuracy"], r.get("mean_forward_probes") or 99))

        return {
            "pareto": rows,
            "eval_split": "full",
            "sample_count": len(dataset),
            "seed": args.seed,
            "device": str(device),
            "insight": "X1 仅 soft_floor 89.3%；验证 hybrid 在全量是否仍优于 auto_route。",
        }

    path = timed_run(run_body, "y17_full419_pareto", "Y17 · 全量 419 Pareto", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase8_result("y17_full419_pareto", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
