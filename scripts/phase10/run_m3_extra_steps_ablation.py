#!/usr/bin/env python3
"""M3 · 多走步是否掉分：fc 步 vs cap 步准确率对比（导师问题 1）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase10_common import CAP, MIN_N, SEED, load_model_bundle, load_test_split, timed_run, write_phase10_result
from evaluate_coconut import expected_answer
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        improved = degraded = unchanged_wrong = unchanged_right = never_fc = 0
        rows = []
        for idx, sample in enumerate(test_set):
            expected = expected_answer(sample, profile)
            fc, preds = first_correct_step(
                model, tokenizer, sample, cap=args.cap, device=device, seed=SEED + idx * 31,
                predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
                expected_fn=expected_answer, eval_profile=profile,
            )
            pred_cap = preds.get(args.cap, preds.get(max(preds), ""))
            if fc is None:
                never_fc += 1
                acc_fc = False
            else:
                pred_fc = preds.get(fc, "")
                acc_fc = pred_fc == expected
            acc_cap = pred_cap == expected
            if fc is None:
                cat = "never_fc"
            elif acc_fc and not acc_cap:
                cat = "degraded_by_extra"
                degraded += 1
            elif not acc_fc and acc_cap:
                cat = "improved_by_extra"
                improved += 1
            elif acc_fc and acc_cap:
                cat = "unchanged_correct"
                unchanged_right += 1
            else:
                cat = "unchanged_wrong"
                unchanged_wrong += 1
            if cat != "never_fc" and cat not in ("degraded_by_extra", "improved_by_extra"):
                pass
            rows.append({"idx": idx, "fc": fc, "acc_at_fc": acc_fc, "acc_at_cap": acc_cap, "category": cat})

        n = len(test_set)
        has_fc = n - never_fc
        return {
            "ablation": {
                "total": n,
                "has_first_correct": has_fc,
                "never_fc": never_fc,
                "unchanged_correct": unchanged_right,
                "improved_by_extra_steps": improved,
                "degraded_by_extra_steps": degraded,
                "unchanged_wrong": unchanged_wrong,
                "pct_improved": round(improved / max(has_fc, 1), 4),
                "pct_degraded": round(degraded / max(has_fc, 1), 4),
                "pct_stop_early_safe": round(unchanged_right / max(has_fc, 1), 4),
            },
            "insight": "若 improved 低、degraded 低 → 适可而止有据；多走步很少救回、很少搞砸。",
            "eval_split": "test_40pct",
            "sample_count": n,
            "device": str(device),
        }

    path = timed_run(run_body, "m3_extra_steps_ablation", "M3 · 多走步 ablation", device=args.device)
    import json
    write_phase10_result("m3_extra_steps_ablation", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
