#!/usr/bin/env python3
"""Y4 · 基于 X3/X4 的可部署单次 forward 规则（n 由 blind_depth 推导）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List

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
from boundary_budget import evaluate_upfront_budget_stop  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from boundary_budget import blind_depth  # noqa: E402


def rule_n_d(sample: dict, cap: int) -> int:
    d = blind_depth(sample)
    return max(1, min(d, cap))


def rule_n_d_minus_1_d4(sample: dict, cap: int) -> int:
    d = blind_depth(sample)
    if d >= 4:
        return max(2, min(d - 1, cap))
    return max(1, min(d, cap))


def rule_n3_if_d4(sample: dict, cap: int) -> int:
    d = blind_depth(sample)
    if d >= 4:
        return 3
    return max(1, min(d, cap))


def rule_n_max2_d_minus1(sample: dict, cap: int) -> int:
    """X4 洞察：fc 常比 blind_depth 早 1–2 步 → 4 跳试 d-1。"""
    d = blind_depth(sample)
    if d >= 4:
        return max(2, min(d - 1, cap))
    if d == 3:
        return 3
    return max(1, min(d, cap))


RULES: List[tuple[str, Callable]] = [
    ("n_eq_d", rule_n_d),
    ("n_d_minus1_if_d4", rule_n_d_minus_1_d4),
    ("n3_if_d4", rule_n3_if_d4),
    ("n_gap_aware_v1", rule_n_max2_d_minus1),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        rows = []
        for name, budget_fn in RULES:
            def make_fn(fn=budget_fn):
                return lambda s, cap=args.cap: fn(s, cap)

            r = evaluate_upfront_budget_stop(
                model,
                tokenizer,
                test_set,
                budget_fn=make_fn(),
                device=device,
                seed=42,
                predict_fn=predict_fn,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
            )
            rows.append(
                {
                    "rule": name,
                    "accuracy": r["accuracy"],
                    "mean_forward_probes": 1.0,
                    "deployable": True,
                }
            )
        rows.sort(key=lambda x: -x["accuracy"])
        return {
            "rules": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
            "insight": "对照 X3：d4 欠预算(Δ=-1)仅 75%；规则应在单次 forward 下逼近 auto_route",
        }

    path = timed_run(run_body, "y4_gap_upfront_rules", "Y4 · gap 感知单次预算规则", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase5_result("y4_gap_upfront_rules", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
