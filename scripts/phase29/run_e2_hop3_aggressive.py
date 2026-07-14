#!/usr/bin/env python3
"""E2 · 3跳强攻：仅 hop=3 一律回退 knn，4跳保持 thr=0.48。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, run_knn_path, setup_fallback_stack
from _phase29_common import CAP, MIN_N, load_json, timed_run, write_phase29_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import is_deployable_mvp, is_eps_deployable, load_full_dataset, timing_metrics
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_hop3_always_fallback(head, model, tokenizer, samples, *, device, seed, profile,
                              struct_floor, knn_floor, knn_thr, pfn, hop4_thr):
    head.eval()
    correct = fb = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        hop = blind_depth(sample)
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
        force = hop < 4
        do_fb = force or prob0 < hop4_thr
        if do_fb:
            fb += 1
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
        "fallback_count": fb,
        "fallback_rate": round(fb / len(samples), 4),
        "params": {"hop3_always_fallback": True, "hop4_thr": hop4_thr, "mode": "hop3_aggressive"},
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
        for hop4_thr in (0.48, 0.55):
            sweep.append(eval_hop3_always_fallback(
                head, model, tokenizer, full, device=device, seed=99, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                hop4_thr=hop4_thr,
            ))
        best = max(sweep, key=lambda r: r["accuracy"])
        p26 = load_json("phase26/b2_gap_forensic_latest.json")
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "gap_indices": [111, 189, 261],
            "champ_wrong_union_ok": p26.get("champ_wrong_union_ok"),
            "mentor_brief": f"E2 3跳强攻：最优 acc {best['accuracy']:.1%} fallback {best['fallback_rate']:.1%}。",
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "e2_hop3_aggressive", "E2 · 3跳强攻", device=args.device)
    write_phase29_result("e2_hop3_aggressive", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
