#!/usr/bin/env python3
"""A3 · 答案分歧仲裁：回退时 struct/knn 答案不同则比置信度。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack
from _phase25_common import load_full_dataset, load_json, timed_run, write_phase25_result
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
        for thr in (0.48, 0.50, 0.52):
            base = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr, answer_arbitrate=False,
            )
            arb = eval_confidence_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr, answer_arbitrate=True,
            )
            variants.append({"thr": thr, "baseline": base, "arbitrate": arb})

        best = max(
            (v["arbitrate"] for v in variants),
            key=lambda r: (r["accuracy"], -(r["fallback_rate"] or 0)),
        )
        p24 = load_json("phase24/y2_confidence_fallback_latest.json")
        y3 = load_json("phase24/y3_error_taxonomy_latest.json")
        return {
            "variants": variants,
            "best": best,
            "full_419": best,
            "baseline_p24": (p24.get("best") or {}).get("accuracy"),
            "union_ceiling": y3.get("baseline_union_acc"),
            "union_miss": y3.get("union_miss"),
            "insight": "P24 Y3：并集缺口 17 题双错；仲裁能否在回退路径多捞 knn_only 5 题。",
            "mentor_brief": (
                f"A3 仲裁：最优 acc {best['accuracy']:.1%} "
                f"arbitrate {best.get('arbitrate_count', 0)} fallback {best['fallback_rate']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "a3_answer_arbitrate", "A3 · 答案仲裁", device=args.device)
    write_phase25_result("a3_answer_arbitrate", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
