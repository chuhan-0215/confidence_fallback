#!/usr/bin/env python3
"""D2 · 跳数分阈值精扫：3跳低 thr + 4跳 0.48。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import run_knn_path, setup_fallback_stack
from _phase28_common import CAP, MIN_N, load_json, timed_run, write_phase28_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import is_deployable_mvp, is_eps_deployable, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_hop_split_thr(head, model, tokenizer, samples, *, device, seed, profile,
                       struct_floor, knn_floor, knn_thr, pfn, hop3_thr, hop4_thr):
    head.eval()
    correct = fallback_count = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        thr = hop3_thr if blind_depth(sample) < 4 else hop4_thr
        n0 = max(MIN_N, min(CAP, struct_floor(sample)))
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
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
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        )
        if prob0 < thr:
            fallback_count += 1
            final, stop_n, _, _ = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=seed + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )
        else:
            final, stop_n = pred0, n0
        if final == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(samples), 4),
        "params": {"hop3_thr": hop3_thr, "hop4_thr": hop4_thr, "mode": "hop_split_thr"},
    }
    row.update(timing_metrics(stop_ns, fcs))
    row["deployable_mvp"] = is_deployable_mvp(row)
    row["eps_deployable"] = is_eps_deployable(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        sweep = []
        for h3 in (0.35, 0.38, 0.40, 0.42, 0.48):
            sweep.append(eval_hop_split_thr(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                hop3_thr=h3, hop4_thr=0.48,
            ))
        best = max(sweep, key=lambda r: (r["accuracy"], -r["fallback_rate"]))
        p25 = load_json("phase25/a1_fallback_finetune_latest.json")
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "baseline_p25": (p25.get("best_thr_row") or {}).get("accuracy"),
            "mentor_brief": f"D2 跳数分阈：最优 3跳thr={best['params']['hop3_thr']} acc {best['accuracy']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "d2_hop_split_thr", "D2 · 跳数分阈", device=args.device)
    write_phase28_result("d2_hop_split_thr", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
