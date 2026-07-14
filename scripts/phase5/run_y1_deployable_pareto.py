#!/usr/bin/env python3
"""Y1 · 可部署策略统一 Pareto（与 Track 28/29 相同 test 40% 划分）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase5_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    make_predict_fn,
    timed_run,
    write_phase5_result,
)
from boundary_budget import evaluate_upfront_budget_stop, make_structure_budget_fn  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from run_auto_submit_experiment import evaluate_policy, make_policies  # noqa: E402
from stop_head_tracks import (  # noqa: E402
    evaluate_hop_split_first_correct_stop,
    evaluate_soft_floor_first_correct_stop,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        policies = make_policies(cap=args.cap)
        rows = []

        for name in ("fixed_3", "auto_route"):
            r = evaluate_policy(
                model, tokenizer, test_set, policies[name], device, cap=args.cap, eval_profile=profile
            )
            rows.append(
                {
                    "strategy": name,
                    "accuracy": r["accuracy"],
                    "mean_forward_probes": 1.0,
                    "deployable": True,
                    "eval_split": "test_40pct",
                    "sample_count": len(test_set),
                }
            )

        for label, fn in (
            ("soft_floor_fc", evaluate_soft_floor_first_correct_stop),
            ("hop_split_fc", evaluate_hop_split_first_correct_stop),
        ):
            kw = dict(
                cap=args.cap,
                min_n=args.min_n,
                device=device,
                seed=42,
                predict_fn=predict_fn,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
            )
            if label == "hop_split_fc":
                kw["split_depth"] = 4
            row = fn(model, tokenizer, test_set, **kw)
            rows.append(
                {
                    "strategy": label,
                    "accuracy": row["accuracy"],
                    "mean_forward_probes": round(row["mean_stop_n"], 3),
                    "stop_timing_acc": row.get("stop_timing_acc"),
                    "deployable": True,
                    "eval_split": "test_40pct",
                    "sample_count": len(test_set),
                }
            )

        budget_fn = make_structure_budget_fn(min_n=1, cap=args.cap)
        up = evaluate_upfront_budget_stop(
            model,
            tokenizer,
            test_set,
            budget_fn=budget_fn,
            device=device,
            seed=42,
            predict_fn=predict_fn,
            expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt,
            eval_profile=profile,
        )
        rows.append(
            {
                "strategy": "blind_depth_once",
                "accuracy": up["accuracy"],
                "mean_forward_probes": 1.0,
                "deployable": True,
                "eval_split": "test_40pct",
                "sample_count": len(test_set),
            }
        )

        rows.sort(key=lambda r: (-r["accuracy"], r["mean_forward_probes"]))
        return {
            "strategies": rows,
            "insight": "与 Track 28/29 相同 test 划分；probe≈mean_stop_n 为 soft/hop 序贯代价",
            "device": str(device),
        }

    path = timed_run(
        run_body,
        "y1_deployable_pareto",
        "Y1 · 可部署策略 Pareto（test 40%）",
        device=args.device,
    )
    # also mirror to phase5 dir
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase5_result("y1_deployable_pareto", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
