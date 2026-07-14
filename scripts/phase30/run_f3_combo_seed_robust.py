#!/usr/bin/env python3
"""F3 · F1 组合方案五种子稳健性 vs 冠军。"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase30"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from _phase30_common import GAP_INDICES, ROBUST_SEEDS, load_json, stats, timed_run, write_phase30_result
from phase23._phase23_common import load_full_dataset
from phase4._phase4_common import load_model_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        f1 = load_json("phase30/f1_combo_hybrid_deadzone_latest.json") or {}
        mod = importlib.import_module("run_f1_combo_hybrid_deadzone")
        eval_fn = mod.eval_combo
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        combo_rows, champ_rows = [], []
        for seed in ROBUST_SEEDS:
            cr = eval_fn(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
                hop4_dead_thr=0.55, hop4_fb_thr=0.48,
            )
            combo_rows.append({"seed": seed, "accuracy": cr["accuracy"], "gap_hit": cr["gap_hit"]})
            fb = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=0.48,
            )
            champ_rows.append(fb["accuracy"])
        cs, chs = stats([r["accuracy"] for r in combo_rows]), stats(champ_rows)
        return {
            "combo_rows": combo_rows,
            "combo_stats": cs,
            "champion_stats": chs,
            "gap_indices": list(GAP_INDICES),
            "f1_seed99": (f1.get("full_419") or {}).get("accuracy"),
            "beat_champion_peak": cs["max"] > 0.9523,
            "mentor_brief": (
                f"F3 组合种子：μ={cs['mean']:.1%} max={cs['max']:.1%} gap3率="
                f"{sum(1 for r in combo_rows if r['gap_hit']>=3)}/5；冠军μ={chs['mean']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "f3_combo_seed_robust", "F3 · 组合种子", device=args.device)
    write_phase30_result("f3_combo_seed_robust", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
