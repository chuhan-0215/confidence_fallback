#!/usr/bin/env python3
"""E1 · P28 最优方案五种子稳健性。"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase28"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from _phase29_common import GAP_INDICES, load_json, timed_run, write_phase29_result
from phase23._phase23_common import load_full_dataset, stats
from phase4._phase4_common import load_model_bundle

ROBUST_SEEDS = (0, 1, 2, 42, 99)
EVAL_DISPATCH = {
    "d2_mid_conf_zone": ("run_d2_mid_conf_zone", "eval_upper_thr_fallback", {"upper_thr": None}),
    "d3_hop_hybrid": ("run_d3_hop_hybrid", "eval_hop_hybrid", {"hop4_thr": 0.48}),
    "d4_always_compare": ("run_d4_always_compare", "eval_always_compare", {}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        p28 = load_json("phase28/d5_gap_closure_rollup_latest.json") or {}
        best_id = (p28.get("best_p28") or {}).get("id") or "d3_hop_hybrid"
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        rows = []
        if best_id in EVAL_DISPATCH:
            mod_name, fn_name, extra = EVAL_DISPATCH[best_id]
            mod = importlib.import_module(mod_name)
            eval_fn = getattr(mod, fn_name)
            if best_id == "d2_mid_conf_zone":
                d2 = load_json("phase28/d2_mid_conf_zone_latest.json") or {}
                extra = {"upper_thr": (d2.get("best") or {}).get("params", {}).get("upper_thr", 0.55)}
            for seed in ROBUST_SEEDS:
                row = eval_fn(
                    head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                    struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, **extra,
                )
                rows.append({"seed": seed, "accuracy": row["accuracy"], "gap_hit": row.get("gap_hit", 0)})
        else:
            for seed in ROBUST_SEEDS:
                row = eval_confidence_fallback(
                    head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                    struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=0.48,
                )
                rows.append({"seed": seed, "accuracy": row["accuracy"], "gap_hit": 0})
        acc_stats = stats([r["accuracy"] for r in rows])
        champ_rows = []
        for seed in ROBUST_SEEDS:
            cr = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=0.48,
            )
            champ_rows.append(cr["accuracy"])
        champ_stats = stats(champ_rows)
        beat = acc_stats["max"] > 0.9523
        return {
            "best_p28_id": best_id,
            "rows": rows,
            "acc_stats": acc_stats,
            "champion_stats": champ_stats,
            "beat_champion_peak": beat,
            "gap_indices": list(GAP_INDICES),
            "mentor_brief": (
                f"E1 {best_id} 五种子 μ={acc_stats['mean']:.1%} max={acc_stats['max']:.1%}；"
                f"冠军 μ={champ_stats['mean']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "e1_best_seed_robust", "E1 · 种子稳健", device=args.device)
    write_phase29_result("e1_best_seed_robust", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
