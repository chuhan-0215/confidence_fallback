#!/usr/bin/env python3
"""V4 · hurt 切片 collateral 复现（baseline vs V3 策略）。"""
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
from _phase34_common import (  # noqa: E402
    HURT_SLICE_IDS,
    PHASE34_OUT,
    TRANSFER_THR,
    eval_agreement_lock,
    eval_tri_zone,
    m2_head_ready,
    write_phase34_result,
)
from dataset_registry import load_slice  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from phase23._phase23_common import CAP, MIN_N  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
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


def load_v3_params() -> tuple[float, float, bool]:
    path = PHASE34_OUT / "v3_hop4_tri_zone_latest.json"
    t_low, t_mid, hop4 = 0.38, 0.46, True
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        t_low = float(data.get("t_low", t_low))
        t_mid = float(data.get("t_mid", t_mid))
    return t_low, t_mid, hop4


@torch.no_grad()
def per_sample_baseline(head, model, tokenizer, sample, *, device, seed, profile,
                        struct_floor, knn_floor, knn_thr, pfn, idx):
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
    final_ok = (pk if triggered else pred0) == expected
    return classify(main_ok, pk == expected if triggered else main_ok, triggered), prob0, pred0, pk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t_low, t_mid, hop4 = load_v3_params()
    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    slice_reports = []
    for sid in HURT_SLICE_IDS:
        meta, samples = load_slice(sid)
        baseline_counts: dict[str, int] = {}
        for idx, sample in enumerate(samples):
            kind, _, _, _ = per_sample_baseline(
                head, model, tokenizer, sample, device=device, seed=args.seed,
                profile=profile, struct_floor=struct_floor, knn_floor=knn_floor,
                knn_thr=knn_thr, pfn=pfn, idx=idx,
            )
            baseline_counts[kind] = baseline_counts.get(kind, 0) + 1

        base_row = eval_confidence_fallback(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR,
        )
        agree_row = eval_agreement_lock(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            hop4_only=False,
        )
        v3_row = eval_tri_zone(
            head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            t_low=t_low, t_mid=t_mid, hop4_only=hop4,
        )
        slice_reports.append({
            "slice_id": sid,
            "label": meta.get("label"),
            "n_samples": len(samples),
            "baseline_per_sample_counts": baseline_counts,
            "policies": {
                "baseline": base_row,
                "agreement_lock": agree_row,
                "hop4_tri_zone": v3_row,
            },
        })
        print(f"{sid}: baseline={base_row['accuracy']:.1%} v3={v3_row['accuracy']:.1%}", flush=True)

    payload = {
        "experiment_id": "v4_collateral_replay",
        "title": "V4 · hurt 切片 collateral 复现",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "v3_params": {"t_low": t_low, "t_mid": t_mid, "hop4_only": hop4},
        "slices": slice_reports,
    }
    write_phase34_result("v4_collateral_replay", payload)
    print(json.dumps(payload["v3_params"], ensure_ascii=False))


if __name__ == "__main__":
    main()
