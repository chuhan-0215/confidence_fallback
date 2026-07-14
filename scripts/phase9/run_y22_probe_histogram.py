#!/usr/bin/env python3
"""Y22 · hybrid probe 成本直方图（按跳数分桶）。"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase9_common import DEPLOY_DEFAULTS, load_model_bundle, load_test_split, timed_run, write_phase9_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from phase8._hybrid_traced import evaluate_hybrid_traced
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step
from phase7._hybrid_eval import first_probe_n, resolve_stop_n


def probe_per_sample(model, tokenizer, samples, *, cap, min_n, device, seed, profile):
    rows = []
    for idx, sample in enumerate(samples):
        d = min(blind_depth(sample), cap)
        sseed = seed + idx * 31
        n0 = first_probe_n("n_eq_d", sample, cap)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=sseed, eval_profile=profile)
        if pred0 == expected_answer(sample, profile):
            probes = 1
            path = "one_shot"
        else:
            fc, preds = first_correct_step(
                model, tokenizer, sample, cap=cap, device=device, seed=sseed,
                predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
                expected_fn=expected_answer, eval_profile=profile,
            )
            stop_n = resolve_stop_n(mode="soft_floor", fc=fc, d=d, min_n=min_n, cap=cap)
            charged = {n0}
            for n in range(max(min_n, 1), stop_n + 1):
                charged.add(n)
            probes = len(charged)
            path = "retry"
        rows.append({"hops": blind_depth(sample), "probes": probes, "path": path})
    return rows


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
        rows = probe_per_sample(
            model, tokenizer, test_set, cap=args.cap, min_n=args.min_n,
            device=device, seed=args.seed, profile=profile,
        )
        all_probes = [r["probes"] for r in rows]
        by_hop: dict = {}
        for r in rows:
            h = str(r["hops"])
            by_hop.setdefault(h, []).append(r["probes"])
        hop_stats = {
            h: {
                "count": len(v),
                "mean_probes": round(statistics.mean(v), 3),
                "p90": sorted(v)[max(0, int(0.9 * len(v)) - 1)],
            }
            for h, v in sorted(by_hop.items())
        }
        sorted_p = sorted(all_probes)
        n = len(sorted_p)
        return {
            "histogram": {
                "mean": round(statistics.mean(all_probes), 3),
                "p50": sorted_p[n // 2],
                "p90": sorted_p[max(0, int(0.9 * n) - 1)],
                "p99": sorted_p[max(0, int(0.99 * n) - 1)],
                "max": max(all_probes),
                "one_shot_rate": round(sum(1 for r in rows if r["path"] == "one_shot") / n, 4),
            },
            "by_hop": hop_stats,
            "eval_split": "test_40pct",
            "sample_count": n,
            "device": str(device),
            "insight": "部署成本模型：p90 probes 与按跳数分桶均值。",
        }

    path = timed_run(run_body, "y22_probe_histogram", "Y22 · probe 直方图", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase9_result("y22_probe_histogram", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
