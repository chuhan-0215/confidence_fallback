#!/usr/bin/env python3
"""U5 · timing 失败模式分析（早停/晚停/从未fc）。"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase19_common import CAP, MIN_N, SEED, load_full_dataset, load_m2_head_state, load_rich_head, timed_run, write_phase19_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import evaluate_rich_stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full_set = load_full_dataset()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        head = load_rich_head(device, load_m2_head_state(device))

        configs = [
            ("phase16_best_acc", 3, 0.5),
            ("phase16_best_timing", 3, 0.35),
            ("phase18_w1", 3, 0.15),
        ]
        reports = []
        for name, min_n, thr in configs:
            from stop_head import first_correct_step
            early = late = hit = never_fc = 0
            delta_hist = Counter()
            for idx, sample in enumerate(full_set):
                expected = expected_answer(sample, profile)
                fc, preds = first_correct_step(
                    model, tokenizer, sample, cap=CAP, device=device, seed=SEED + idx * 31,
                    predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
                )
                if fc is None:
                    never_fc += 1
                    continue
                row_n = None
                prev_pred = ""
                streak = 0
                from stop_head import _rich_step_features, extract_latent_hidden
                stop_n = CAP
                for n in range(1, CAP + 1):
                    pred = preds[n]
                    answer_bucket, streak, changed = _rich_step_features(pred, prev_pred, streak)
                    prev_pred = pred
                    prompt = build_eval_prompt(sample, n, seed=SEED + idx * 31 + n,
                        choice_order=profile.choice_order,
                        shuffle_edges=profile.prompt_mode != "fixed_edges")
                    input_ids = __import__("torch").tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
                    hidden = extract_latent_hidden(model, input_ids, pass_idx=n - 1).to(device)
                    prob = float(__import__("torch").sigmoid(head(
                        hidden.unsqueeze(0),
                        __import__("torch").tensor([n], device=device),
                        __import__("torch").tensor([answer_bucket], device=device),
                        __import__("torch").tensor([streak], device=device),
                        __import__("torch").tensor([changed], device=device),
                    )).item())
                    stop_n = n
                    if n >= min_n and prob >= thr:
                        break
                if stop_n == fc:
                    hit += 1
                elif stop_n < fc:
                    early += 1
                    delta_hist[fc - stop_n] += 1
                else:
                    late += 1
                    delta_hist[stop_n - fc] += 1

            total_fc = hit + early + late
            reports.append({
                "config": name, "min_n": min_n, "threshold": thr,
                "timing_hit": hit, "early_stop": early, "late_stop": late, "never_fc": never_fc,
                "timing_acc": round(hit / total_fc, 4) if total_fc else 0,
                "early_pct": round(early / total_fc, 4) if total_fc else 0,
                "late_pct": round(late / total_fc, 4) if total_fc else 0,
                "delta_histogram": dict(delta_hist),
            })

        return {
            "reports": reports,
            "insight": "若 early_stop 主导 → 需 patience/延后；若 late_stop 主导 → 需更激进 thr 或 fc 标签。",
            "device": str(device),
        }

    path = timed_run(run_body, "u5_failure_analysis", "U5 · 失败分析", device=args.device)
    import json
    write_phase19_result("u5_failure_analysis", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
