#!/usr/bin/env python3
"""A2 · 4跳专项回退：仅 hop≥4 低置信时回退 knn。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from _phase25_common import FINE_FALLBACK, load_full_dataset, load_json, timed_run, write_phase25_result
from phase23._phase23_common import filter_by_hop


from phase4._phase4_common import load_model_bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        hop3 = filter_by_hop(full, 3)
        hop4 = filter_by_hop(full, 4)
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

        sweep = []
        for thr in FINE_FALLBACK:
            row = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr, hop4_only=True,
            )
            sweep.append(row)

        best = max(sweep, key=lambda r: (r["accuracy"], -(r["fallback_rate"] or 0)))
        thr = best["params"]["fallback_thr"]
        hop3_row = eval_confidence_fallback(
            head, model, tokenizer, hop3, device=device, seed=99, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=thr, hop4_only=True,
        )
        hop4_row = eval_confidence_fallback(
            head, model, tokenizer, hop4, device=device, seed=99, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=thr, hop4_only=True,
        )
        p24 = load_json("phase24/y2_confidence_fallback_latest.json")
        return {
            "sweep": sweep,
            "best": best,
            "hop3_slice": hop3_row,
            "hop4_slice": hop4_row,
            "full_419": best,
            "baseline_p24_global": (p24.get("best") or {}).get("accuracy"),
            "insight": "Y3：4跳 knn 多对 5 题；4跳专项回退能否保全 3跳同时抬全量 acc。",
            "mentor_brief": (
                f"A2 4跳回退：thr={thr} 全量 {best['accuracy']:.1%} "
                f"fallback {best['fallback_rate']:.1%}；3跳 {hop3_row['accuracy']:.1%} 4跳 {hop4_row['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "a2_hop4_fallback", "A2 · 4跳回退", device=args.device)
    write_phase25_result("a2_hop4_fallback", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
