#!/usr/bin/env python3
"""Z2 · min_n=1 审计：为何 acc 98.81% 高于 oracle 97.62%。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase6_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    make_predict_fn,
    timed_run,
    write_phase6_result,
)
from boundary_budget import blind_depth  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt, reasoning_hops  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import first_correct_step  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = lambda s, n, seed: predict_at_n(
            model, tokenizer, s, n, device, seed=seed, eval_profile=profile
        )

        oracle_correct = timing_hits = timing_total = policy_correct = 0
        suspicious = []
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

            # oracle: stop at fc if exists else d
            if fc is not None:
                oracle_stop = fc
                oracle_pred = preds.get(fc, "")
                timing_total += 1
                if oracle_stop == fc:
                    timing_hits += 1
            else:
                oracle_stop = min(d, args.cap)
                oracle_pred = preds.get(oracle_stop, "")

            if oracle_pred == expected:
                oracle_correct += 1

            # soft_floor min_n=1
            if fc is not None and fc >= args.min_n:
                stop_n = fc
            else:
                stop_n = max(args.min_n, min(d, args.cap))
            final_pred = preds.get(stop_n, preds.get(args.cap, ""))
            ok = final_pred == expected
            if ok:
                policy_correct += 1
            else:
                failures.append(
                    {
                        "idx": idx,
                        "hops": hops,
                        "blind_depth": d,
                        "first_correct": fc,
                        "stop_n": stop_n,
                        "expected": expected,
                        "got": final_pred,
                    }
                )

            # suspicious: policy correct but oracle wrong, or fc=1 with high timing
            if ok and oracle_pred != expected:
                suspicious.append({"idx": idx, "reason": "policy_ok_oracle_fail", "fc": fc, "stop_n": stop_n})
            if fc == 1 and ok and d >= 3:
                suspicious.append({"idx": idx, "reason": "fc1_early_on_deep", "hops": hops, "d": d})

        n = len(test_set)
        return {
            "min_n": args.min_n,
            "seed": args.seed,
            "sample_count": n,
            "oracle_first_correct": {
                "accuracy": round(oracle_correct / n, 4),
                "correct": oracle_correct,
                "timing_acc": round(timing_hits / timing_total, 4) if timing_total else None,
            },
            "soft_floor_policy": {
                "accuracy": round(policy_correct / n, 4),
                "correct": policy_correct,
            },
            "failures": failures[:20],
            "failure_count": len(failures),
            "suspicious": suspicious[:30],
            "suspicious_count": len(suspicious),
            "insight": (
                "min_n=1 时 fc≥1 即停，timing 定义下 stop_n==fc 比例极高；"
                "若 acc>oracle 需检查 oracle 定义（oracle 是否含 fc=None 样本）。"
            ),
            "device": str(device),
        }

    path = timed_run(run_body, "z2_min_n1_audit", "Z2 · min_n=1 审计", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result("z2_min_n1_audit", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
