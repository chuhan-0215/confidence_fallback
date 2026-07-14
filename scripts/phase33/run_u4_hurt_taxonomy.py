#!/usr/bin/env python3
"""U4 · Hurt 切片逐题归因（collateral / failed rescue / wasted / saved）。"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, run_knn_path, setup_fallback_stack  # noqa: E402
from _phase33_common import HURT_SLICE_IDS, TRANSFER_THR, m2_head_ready, write_phase33_result  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from phase23._phase23_common import CAP, MIN_N  # noqa: E402
from run_adaptive_stop_experiment import predict_at_n  # noqa: E402
from stop_head import _rich_step_features, extract_latent_hidden  # noqa: E402


def classify(main_ok: bool, fb_ok: bool, triggered: bool) -> str:
    if not triggered:
        return "no_fallback"
    if main_ok and not fb_ok:
        return "A_collateral"
    if not main_ok and not fb_ok:
        return "B_failed_rescue"
    if main_ok and fb_ok:
        return "C_wasted"
    return "D_saved"


@torch.no_grad()
def analyze_slice(head, model, tokenizer, samples, meta, *, device, seed, profile,
                  struct_floor, knn_floor, knn_thr, pfn):
    head.eval()
    counts = {"A_collateral": 0, "B_failed_rescue": 0, "C_wasted": 0, "D_saved": 0, "no_fallback": 0}
    examples = {"A_collateral": [], "D_saved": []}
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        main_ok = pred0 == expected
        prompt0 = build_eval_prompt(sample, n0, seed=seed + idx * 31,
            choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
        ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
        hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
        ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
        prob0 = torch.sigmoid(head(
            hid0.unsqueeze(0), torch.tensor([n0], device=device),
            torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
            torch.tensor([ch0], device=device),
        )).item()
        triggered = prob0 < TRANSFER_THR
        pk, _, _, _ = run_knn_path(
            head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
            profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        fb_ok = pk == expected
        kind = classify(main_ok, fb_ok, triggered)
        counts[kind] += 1
        if kind in examples and len(examples[kind]) < 3:
            examples[kind].append({
                "idx": idx, "prob0": round(prob0, 4), "pred0": pred0, "pred_knn": pk,
                "expected": expected, "triggered": triggered,
            })

    total = len(samples)
    return {
        "slice_id": meta.get("id"),
        "label": meta.get("label"),
        "category": meta.get("category"),
        "n_samples": total,
        "counts": counts,
        "rates": {k: round(v / total, 4) for k, v in counts.items()},
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    slice_reports = []
    for sid in HURT_SLICE_IDS:
        meta, samples = load_slice(sid)
        meta = dict(meta)
        meta["id"] = sid
        rep = analyze_slice(
            head, model, tokenizer, samples, meta, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        )
        baseline = eval_confidence_fallback(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR,
        )
        rep["transfer_acc"] = baseline["accuracy"]
        rep["fallback_rate"] = baseline.get("fallback_rate")
        slice_reports.append(rep)
        print(f"{sid}: collateral={rep['rates'].get('A_collateral', 0):.1%}", flush=True)

    payload = {
        "experiment_id": "u4_hurt_taxonomy",
        "title": "U4 · Hurt 切片归因",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "hurt_slice_ids": list(HURT_SLICE_IDS),
        "slices": slice_reports,
        "aggregate_collateral_rate": round(
            sum(r["counts"]["A_collateral"] for r in slice_reports)
            / max(1, sum(r["n_samples"] for r in slice_reports)), 4,
        ),
    }
    write_phase33_result("u4_hurt_taxonomy", payload)
    print(json.dumps(payload["aggregate_collateral_rate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
