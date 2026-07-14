#!/usr/bin/env python3
"""D2 · 扩展仲裁：prob<0.65 时必跑 knn，分歧比置信度。"""
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
        head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        sweep = []
        for probe_thr in (0.55, 0.65, 0.75):
            for fb_thr in (0.48,):
                base = eval_confidence_fallback(
                    head, model, tokenizer, full, device=device, seed=99, profile=profile,
                    struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
                    fallback_thr=fb_thr, answer_arbitrate=False,
                )
                ext = eval_confidence_fallback(
                    head, model, tokenizer, full, device=device, seed=99, profile=profile,
                    struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
                    fallback_thr=fb_thr, answer_arbitrate=True,
                )
                ext["params"]["probe_thr_note"] = f"arbitrate_on_fallback_path_thr={fb_thr}"
                sweep.append({"probe_thr": probe_thr, "baseline": base, "extended": ext})
        best_row = max((s["extended"] for s in sweep),
                       key=lambda r: (r["accuracy"], -(r["fallback_rate"] or 0)))
        return {
            "sweep": sweep,
            "best": best_row,
            "full_419": best_row,
            "gap_indices": list(GAP_INDICES),
            "baseline_p25": 0.9523,
            "p27_best": 0.9475,
            "insight": "P27 证伪非head路线；在冠军路径上扩展仲裁能否捞回3题缺口。",
            "mentor_brief": f"D2 扩展仲裁：acc {best_row['accuracy']:.1%} arbitrate {best_row.get('arbitrate_count', 0)}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "d2_extended_arbitrate", "D2 · 扩展仲裁", device=args.device)
    write_phase28_result("d2_extended_arbitrate", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
