#!/usr/bin/env python3
"""D1 · 缺口三题靶向诊断：idx 111/189/261 为何 fallback 未捞回。"""
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
from _phase28_common import GAP_INDICES, SEED, load_json, timed_run, write_phase28_result
from boundary_budget import blind_depth
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase23._phase23_common import load_full_dataset
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden


@torch.no_grad()
def diagnose_sample(head, model, tokenizer, sample, idx, *, device, seed, profile,
                    struct_floor, knn_floor, knn_thr, pfn, thr):
    expected = expected_answer(sample, profile)
    hop = blind_depth(sample)
    n0 = max(MIN_N, min(CAP, struct_floor(sample)))
    pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed, eval_profile=profile)
    prompt0 = build_eval_prompt(sample, n0, seed=seed,
        choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
    ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
    hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
    ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
    prob0 = torch.sigmoid(head(
        hid0.unsqueeze(0), torch.tensor([n0], device=device),
        torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
        torch.tensor([ch0], device=device),
    )).item()
    pk, nk, _, pk_prob = run_knn_path(
        head, model, tokenizer, sample, device=device, seed=seed,
        profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
    )
    triggered = prob0 < thr
    champ_pred = pk if triggered else pred0
    return {
        "idx": idx, "hop": hop, "expected": expected,
        "struct_pred": pred0, "struct_ok": pred0 == expected,
        "knn_pred": pk, "knn_ok": pk == expected,
        "prob0": round(prob0, 4), "fallback_thr": thr,
        "fallback_triggered": triggered,
        "champion_pred": champ_pred, "champion_ok": champ_pred == expected,
        "knn_stop_n": nk, "knn_final_prob": round(pk_prob, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        thr = 0.48
        cases = []
        for idx in GAP_INDICES:
            cases.append(diagnose_sample(
                head, model, tokenizer, full[idx], idx, device=device, seed=SEED + idx * 31,
                profile=profile, struct_floor=struct_floor, knn_floor=knn_floor,
                knn_thr=knn_thr, pfn=pfn, thr=thr,
            ))
        not_triggered = sum(1 for c in cases if not c["fallback_triggered"])
        return {
            "cases": cases,
            "gap_indices": list(GAP_INDICES),
            "not_triggered_count": not_triggered,
            "insight": "P26 B2：3题均 knn_ok struct_wrong；诊断 fallback 是否未触发。",
            "mentor_brief": (
                f"D1 三题诊断：未触发 {not_triggered}/3；"
                f"详情见 cases。"
            ),
            "sample_count": 3,
            "device": str(device),
        }

    path = timed_run(run_body, "d1_gap3_diagnosis", "D1 · 三题诊断", device=args.device)
    write_phase28_result("d1_gap3_diagnosis", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
