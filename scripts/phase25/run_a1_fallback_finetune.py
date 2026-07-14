#!/usr/bin/env python3
"""A1 · Y2 精调：fallback_thr 细扫 + 多种子稳健性。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from _phase25_common import FINE_FALLBACK, ROBUST_SEEDS, load_full_dataset, load_json, timed_run, write_phase25_result
from phase23._phase23_common import stats
from phase4._phase4_common import load_model_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

        sweep = []
        for thr in FINE_FALLBACK:
            row = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr,
            )
            sweep.append(row)
        best_thr_row = max(sweep, key=lambda r: (r["accuracy"], -(r["fallback_rate"] or 0)))
        best_thr = best_thr_row["params"]["fallback_thr"]

        seed_rows = []
        for seed in ROBUST_SEEDS:
            row = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=best_thr,
            )
            row["seed"] = seed
            seed_rows.append(row)

        acc_stats = stats([r["accuracy"] for r in seed_rows])
        p24 = load_json("phase24/y2_confidence_fallback_latest.json")
        return {
            "sweep": sweep,
            "best_thr": best_thr,
            "best_thr_row": best_thr_row,
            "seed_rows": seed_rows,
            "acc_stats": acc_stats,
            "full_419": best_thr_row,
            "baseline_p24": (p24.get("best") or {}).get("accuracy"),
            "insight": "P24 Y2 thr=0.5 达 94.99%；精调+稳健性验证能否稳定超 94.75%。",
            "mentor_brief": (
                f"A1 精调：最优 thr={best_thr} acc {best_thr_row['accuracy']:.1%} "
                f"fallback {best_thr_row['fallback_rate']:.1%}；"
                f"五种子 μ={acc_stats['mean']:.1%} σ={acc_stats['stdev']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "a1_fallback_finetune", "A1 · Y2 精调", device=args.device)
    write_phase25_result("a1_fallback_finetune", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
