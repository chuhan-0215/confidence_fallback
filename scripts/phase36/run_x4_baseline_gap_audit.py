#!/usr/bin/env python3
"""X4 · baseline 缺口审计：P25 95.23% vs 当前 94.75% 逐题对比。"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR, _full_knn, _main_step  # noqa: E402
from _phase36_common import m2_head_ready, write_phase36_result  # noqa: E402
from evaluate_coconut import expected_answer  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


@torch.no_grad()
def per_question_baseline(head, model, tokenizer, samples, *, device, seed, profile, struct_floor, knn_floor, knn_thr, pfn):
    from boundary_budget import blind_depth

    rows = []
    correct = 0
    for idx, sample in enumerate(samples):
        n0, pred0, prob0, expected = _main_step(
            head, model, tokenizer, sample, device=device, seed=seed,
            profile=profile, struct_floor=struct_floor, idx=idx,
        )
        final = pred0
        used_fb = False
        if prob0 < TRANSFER_THR:
            used_fb = True
            pk, _, _ = _full_knn(
                head, model, tokenizer, sample, device=device, seed=seed,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn, idx=idx,
            )
            final = pk
        ok = final == expected
        correct += int(ok)
        rows.append({
            "idx": idx,
            "n0": n0,
            "blind_depth": blind_depth(sample),
            "prob0": round(prob0, 4),
            "pred0": pred0,
            "final": final,
            "expected": expected,
            "used_fallback": used_fb,
            "correct": ok,
        })
    return rows, correct / len(samples) if samples else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
    samples = load_full_dataset()

    per_q, acc_manual = per_question_baseline(
        head, model, tokenizer, samples, device=device, seed=args.seed,
        profile=profile, struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn,
    )
    batch = eval_confidence_fallback(
        head, model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
        struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, fallback_thr=TRANSFER_THR,
    )

    wrong = [r for r in per_q if not r["correct"]]
    fb_wrong = [r for r in wrong if r["used_fallback"]]
    main_wrong = [r for r in wrong if not r["used_fallback"]]

    payload = {
        "experiment_id": "x4_baseline_gap_audit",
        "title": "X4 · baseline 缺口审计",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "p25_reference_acc": 0.9523,
        "current_batch_acc": batch["accuracy"],
        "current_manual_acc": round(acc_manual, 4),
        "fallback_count": batch.get("fallback_count"),
        "wrong_count": len(wrong),
        "fallback_wrong_count": len(fb_wrong),
        "main_wrong_count": len(main_wrong),
        "wrong_samples": wrong[:20],
        "fallback_wrong_samples": fb_wrong[:12],
        "insight": "若 fallback_count 与 P25 相同但 acc 更低 → 同触发不同结果（kNN 路径漂移）",
    }
    write_phase36_result("x4_baseline_gap_audit", payload)
    print(json.dumps({
        "batch_acc": batch["accuracy"],
        "wrong": len(wrong),
        "fb_wrong": len(fb_wrong),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
