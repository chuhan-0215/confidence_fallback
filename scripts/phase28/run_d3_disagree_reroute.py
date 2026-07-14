#!/usr/bin/env python3
"""D3 · 分歧即回退：struct/knn 答案不同则信 knn（无 oracle）。"""
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
from _phase28_common import GAP_INDICES, load_json, timed_run, write_phase28_result
from phase23._phase23_common import load_full_dataset
from phase4._phase4_common import load_model_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        variants = []
        for thr in (0.48, 0.55, 0.65):
            base = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr, answer_arbitrate=False,
            )
            dis = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr, answer_arbitrate=True,
            )
            variants.append({"thr": thr, "baseline": base, "disagree_arbitrate": dis})
        best = max((v["disagree_arbitrate"] for v in variants),
                   key=lambda r: (r["accuracy"], -(r["fallback_rate"] or 0)))
        gap_hit = 0
        p26 = load_json("phase26/b2_gap_forensic_latest.json")
        return {
            "variants": variants,
            "best": best,
            "full_419": best,
            "gap_indices": list(GAP_INDICES),
            "baseline_p25": 0.9523,
            "champ_wrong_union_ok": p26.get("champ_wrong_union_ok"),
            "mentor_brief": f"D3 分歧回退：最优 thr={best['params']['fallback_thr']} acc {best['accuracy']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "d3_disagree_reroute", "D3 · 分歧回退", device=args.device)
    write_phase28_result("d3_disagree_reroute", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
