#!/usr/bin/env python3
"""W4 · hurt 切片 collateral 复现（含 combo 策略）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import eval_agreement_lock, eval_tri_zone  # noqa: E402
from _phase35_common import (  # noqa: E402
    HURT_SLICE_IDS,
    PHASE35_OUT,
    TRANSFER_THR,
    eval_agreement_tri_zone,
    load_p34_best,
    m2_head_ready,
    write_phase35_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def load_w2_best() -> tuple[float, float]:
    path = PHASE35_OUT / "w2_combo_grid_sweep_latest.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        best = data.get("best_val") or {}
        if data.get("best_cross_score", -999) > -900:
            top = (data.get("top5_cross") or [{}])[0]
            return float(top.get("t_low", best.get("t_low", 0.4))), float(top.get("t_mid", best.get("t_mid", 0.48)))
        return float(best.get("t_low", 0.4)), float(best.get("t_mid", 0.48))
    return load_p34_best()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t_low, t_mid = load_w2_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    slices_out = []
    for sid in HURT_SLICE_IDS:
        meta, samples = load_slice(sid)
        if not samples:
            continue
        baseline = eval_confidence_fallback(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR,
        )
        agreement = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR, hop4_only=False,
        )
        tri_zone = eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=False,
        )
        combo = eval_agreement_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=False,
        )
        slices_out.append({
            "slice_id": sid,
            "label": meta.get("label"),
            "n_samples": len(samples),
            "params": {"t_low": t_low, "t_mid": t_mid},
            "policies": {
                "baseline": baseline,
                "agreement_lock": agreement,
                "tri_zone": tri_zone,
                "agreement_tri_zone": combo,
            },
        })
        print(f"{sid} combo={combo['accuracy']:.2%} baseline={baseline['accuracy']:.2%}", flush=True)

    payload = {
        "experiment_id": "w4_collateral_replay",
        "title": "W4 · hurt 切片 combo 复现",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "params": {"t_low": t_low, "t_mid": t_mid},
        "slices": slices_out,
    }
    write_phase35_result("w4_collateral_replay", payload)
    print(json.dumps({"n_slices": len(slices_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
