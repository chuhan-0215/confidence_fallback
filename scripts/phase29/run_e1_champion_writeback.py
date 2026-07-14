#!/usr/bin/env python3
"""E1 · 冠军+写回融合：zero_after4 写回 + confidence_fallback thr=0.48。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from _phase29_common import load_json, timed_run, write_phase29_result
from coconut_feedback import apply_feedback_config, default_feedback_strategies
from phase23._phase23_common import load_full_dataset
from phase4._phase4_common import load_model_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        strat = next(s for s in default_feedback_strategies() if s["id"] == "zero_after4")
        apply_feedback_config(model, strat)
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        row = eval_confidence_fallback(
            head, model, tokenizer, full, device=device, seed=99, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=0.48,
        )
        row["params"]["feedback_id"] = "zero_after4"
        row["params"]["mode"] = "champion_plus_writeback"
        p27 = load_json("phase27/c2_writeback_structure_latest.json")
        return {
            "full_419": row,
            "baseline_p25": 0.9523,
            "baseline_c2": (p27.get("best") or {}).get("accuracy"),
            "mentor_brief": f"E1 冠军+写回：acc {row['accuracy']:.1%} feedback zero_after4。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "e1_champion_writeback", "E1 · 冠军+写回", device=args.device)
    write_phase29_result("e1_champion_writeback", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
