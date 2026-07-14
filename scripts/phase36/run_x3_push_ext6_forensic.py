#!/usr/bin/env python3
"""X3 · push_ext6 校准法医：prob0 分布 + 各策略逐题。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(ROOT / "scripts" / "phase35"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR, _main_step, eval_agreement_lock, eval_tri_zone  # noqa: E402
from _phase35_common import eval_agreement_tri_zone  # noqa: E402
from _phase36_common import P34_TRI_ZONE, load_w2_best, m2_head_ready, write_phase36_result  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


@torch.no_grad()
def per_sample_forensic(head, model, tokenizer, sample, *, device, seed, profile, struct_floor, knn_floor, idx):
    from _phase34_common import _knn_preview, _full_knn

    n0, pred0, prob0, expected = _main_step(
        head, model, tokenizer, sample, device=device, seed=seed,
        profile=profile, struct_floor=struct_floor, idx=idx,
    )
    pred_prev = _knn_preview(
        model, tokenizer, sample, device=device, seed=seed,
        profile=profile, knn_floor=knn_floor, idx=idx,
    )
    correct_main = pred0 == expected
    agree = pred0 == pred_prev
    return {
        "n0": n0,
        "prob0": round(prob0, 4),
        "pred0": pred0,
        "pred_knn_preview": pred_prev,
        "expected": expected,
        "correct_main": correct_main,
        "agree": agree,
        "below_tau": prob0 < TRANSFER_THR,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    w2_low, w2_mid = load_w2_best()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    target_slices = ("push_ext6_from4", "push_ext5_from3", "v_diamond_5")
    slices_out = []
    for sid in target_slices:
        meta, samples = load_slice(sid)
        if not samples:
            continue
        policies = {
            "baseline": eval_confidence_fallback(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
            ),
            "agreement_lock": eval_agreement_lock(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
            ),
            "tri_zone": eval_tri_zone(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
                t_low=P34_TRI_ZONE[0], t_mid=P34_TRI_ZONE[1],
            ),
            "combo_w2": eval_agreement_tri_zone(
                head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
                struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, t_low=w2_low, t_mid=w2_mid,
            ),
        }
        forensics = [per_sample_forensic(
            head, model, tokenizer, s, device=device, seed=args.seed,
            profile=profile, struct_floor=sf, knn_floor=kf, idx=i,
        ) for i, s in enumerate(samples)]
        prob_bins = {"<0.38": 0, "0.38-0.48": 0, ">=0.48": 0}
        for f in forensics:
            p = f["prob0"]
            if p < 0.38:
                prob_bins["<0.38"] += 1
            elif p < 0.48:
                prob_bins["0.38-0.48"] += 1
            else:
                prob_bins[">=0.48"] += 1
        slices_out.append({
            "slice_id": sid,
            "label": meta.get("label"),
            "n_samples": len(samples),
            "prob_bins": prob_bins,
            "agree_rate": round(sum(1 for f in forensics if f["agree"]) / len(forensics), 4),
            "below_tau_rate": round(sum(1 for f in forensics if f["below_tau"]) / len(forensics), 4),
            "policies": policies,
            "forensics_sample": forensics[:8],
        })
        print(f"{sid} baseline={policies['baseline']['accuracy']:.2%} combo_w2={policies['combo_w2']['accuracy']:.2%}", flush=True)

    payload = {
        "experiment_id": "x3_push_ext6_forensic",
        "title": "X3 · push 系列校准法医",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "w2_params": {"t_low": w2_low, "t_mid": w2_mid},
        "slices": slices_out,
    }
    write_phase36_result("x3_push_ext6_forensic", payload)
    print(json.dumps({"n": len(slices_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
