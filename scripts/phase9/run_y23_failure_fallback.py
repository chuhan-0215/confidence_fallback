#!/usr/bin/env python3
"""Y23 · hybrid 错题回退矩阵：失败样本上 auto_route / soft_floor 能否救回。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase9_common import DEPLOY_DEFAULTS, load_model_bundle, load_test_split, timed_run, write_phase9_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase5._phase5_common import make_predict_fn
from phase8._hybrid_traced import evaluate_hybrid_traced
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head_tracks import evaluate_soft_floor_first_correct_stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=DEPLOY_DEFAULTS["cap"])
    ap.add_argument("--min-n", type=int, default=DEPLOY_DEFAULTS["min_n"])
    ap.add_argument("--seed", type=int, default=DEPLOY_DEFAULTS["seed"])
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        hybrid = evaluate_hybrid_traced(
            model, tokenizer, test_set, cap=args.cap, min_n=args.min_n,
            device=device, seed=args.seed, profile=profile,
        )
        policies = make_policies(cap=args.cap)
        fallback_rows = []
        rescued = {"auto_route": 0, "soft_floor_fc": 0}
        for fail in hybrid["failures"]:
            idx = fail["idx"]
            sample = test_set[idx]
            expected = expected_answer(sample, profile)
            ar = evaluate_policy(model, tokenizer, [sample], policies["auto_route"], device, cap=args.cap, eval_profile=profile)
            ar_ok = ar["correct"] == 1
            sf = evaluate_soft_floor_first_correct_stop(
                model, tokenizer, [sample], cap=args.cap, min_n=args.min_n,
                device=device, seed=args.seed, predict_fn=predict_fn,
                expected_fn=expected_answer, build_prompt_fn=build_eval_prompt, eval_profile=profile,
            )
            sf_ok = sf["correct"] == 1
            if ar_ok:
                rescued["auto_route"] += 1
            if sf_ok:
                rescued["soft_floor_fc"] += 1
            fallback_rows.append({
                "idx": idx,
                "category": fail["category"],
                "hybrid_got": fail["got"],
                "expected": expected,
                "auto_route_ok": ar_ok,
                "soft_floor_ok": sf_ok,
            })
        return {
            "hybrid_wrong": hybrid["wrong_count"],
            "fallback_matrix": fallback_rows,
            "rescued_if_switched": rescued,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
            "insight": "5 题错题是否有回退价值；通常 hybrid 已 dominate，预期 rescued≈0。",
        }

    path = timed_run(run_body, "y23_failure_fallback", "Y23 · 失败回退矩阵", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase9_result("y23_failure_fallback", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
