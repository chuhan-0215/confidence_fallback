#!/usr/bin/env python3
"""B2 · 缺口法医：冠军方案错而并集对的题。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase26_common import SEED, load_full_dataset, load_json, timed_run, write_phase26_result
from _fallback_eval import eval_confidence_fallback, run_knn_path, setup_fallback_stack
from boundary_budget import blind_depth, make_structure_budget_fn
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden


@torch.no_grad()
def predict_struct(model, tokenizer, sample, device, seed, profile, struct_floor):
    n = max(3, min(8, struct_floor(sample)))
    pred = predict_at_n(model, tokenizer, sample, n, device, seed=seed, eval_profile=profile)
    return pred, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)
        struct_floor_fn = make_structure_budget_fn(min_n=3, cap=8)
        thr = 0.48

        champ_wrong_union_ok = champ_wrong_both_wrong = 0
        by_hop = {"3": 0, "4": 0}
        gap_ids = []

        for idx, sample in enumerate(full):
            expected = expected_answer(sample, profile)
            # champion path (simplified inline)
            n0 = max(3, min(8, struct_floor(sample)))
            pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=SEED + idx * 31, eval_profile=profile)
            prompt0 = build_eval_prompt(sample, n0, seed=SEED + idx * 31,
                choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
            ids0 = torch.tensor([tokenizer.encode(prompt0, add_special_tokens=False)], device=device)
            hid0 = extract_latent_hidden(model, ids0, pass_idx=n0 - 1).to(device)
            ab0, st0, ch0 = _rich_step_features(pred0, "", 1)
            prob0 = torch.sigmoid(head(
                hid0.unsqueeze(0), torch.tensor([n0], device=device),
                torch.tensor([ab0], device=device), torch.tensor([st0], device=device),
                torch.tensor([ch0], device=device),
            )).item()
            if prob0 < thr:
                champ_pred, _ = run_knn_path(
                    head, model, tokenizer, sample, device=device, seed=SEED + idx * 31,
                    profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                )[:2]
            else:
                champ_pred = pred0
            ps, _ = predict_struct(model, tokenizer, sample, device, SEED + idx * 31, profile, struct_floor_fn)
            pk, _ = run_knn_path(
                head, model, tokenizer, sample, device=device, seed=SEED + idx * 31,
                profile=profile, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            )[:2]
            union_ok = ps == expected or pk == expected
            champ_ok = champ_pred == expected
            hop = str(blind_depth(sample))
            if not champ_ok and union_ok:
                champ_wrong_union_ok += 1
                by_hop[hop] = by_hop.get(hop, 0) + 1
                gap_ids.append({"idx": idx, "hop": hop, "struct_ok": ps == expected, "knn_ok": pk == expected})
            if not champ_ok and not union_ok:
                champ_wrong_both_wrong += 1

        p24 = load_json("phase24/y3_error_taxonomy_latest.json")
        p25 = load_json("phase25/a1_fallback_finetune_latest.json")
        return {
            "champ_wrong_union_ok": champ_wrong_union_ok,
            "champ_wrong_both_wrong": champ_wrong_both_wrong,
            "by_hop": by_hop,
            "gap_samples": gap_ids[:20],
            "union_ceiling": p24.get("baseline_union_acc"),
            "champion_acc": (p25.get("best_thr_row") or {}).get("accuracy"),
            "insight": "95.2% 距并集 95.94% 差 ~3 题；定位冠军错但并集对的缺口。",
            "mentor_brief": (
                f"B2 缺口：冠军错并集对 {champ_wrong_union_ok} 题；"
                f"冠军错双错 {champ_wrong_both_wrong} 题。"
            ),
            "sample_count": len(full),
            "device": str(device),
        }

    path = timed_run(run_body, "b2_gap_forensic", "B2 · 缺口法医", device=args.device)
    write_phase26_result("b2_gap_forensic", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
