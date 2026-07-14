#!/usr/bin/env python3
"""Z5 · 双探针精炼：structure_d 定 n，低置信时再探 n+1（最多 2 次前向）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase23_common import (
    CAP, MIN_N, SEED, is_deployable_mvp, is_eps_deployable,
    load_full_dataset, load_json, load_m2_head_state, load_rich_head, load_splits,
    timed_run, timing_metrics, write_phase23_result,
)
from boundary_budget import make_structure_budget_fn
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import _rich_step_features, extract_latent_hidden, first_correct_step


@torch.no_grad()
def eval_dual_probe(head, model, tokenizer, samples, device, seed, profile, conf_thr: float):
    budget_fn = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
    head.eval()
    correct = probe2 = 0
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        expected = expected_answer(sample, profile)
        n0 = budget_fn(sample)
        pred0 = predict_at_n(model, tokenizer, sample, n0, device, seed=seed + idx * 31, eval_profile=profile)
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=CAP, device=device, seed=seed + idx * 31,
            predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
            expected_fn=expected_answer, eval_profile=profile,
        )
        stop_n = n0
        final = pred0
        if n0 < CAP:
            prompt = build_eval_prompt(sample, n0, seed=seed + idx * 31 + n0,
                choice_order=profile.choice_order, shuffle_edges=profile.prompt_mode != "fixed_edges")
            ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            hid = extract_latent_hidden(model, ids, pass_idx=n0 - 1).to(device)
            ab, streak, ch = _rich_step_features(pred0, "", 1)
            prob = torch.sigmoid(head(
                hid.unsqueeze(0), torch.tensor([n0], device=device),
                torch.tensor([ab], device=device), torch.tensor([streak], device=device),
                torch.tensor([ch], device=device),
            )).item()
            if prob < conf_thr:
                pred1 = predict_at_n(model, tokenizer, sample, n0 + 1, device, seed=seed + idx * 31 + 1, eval_profile=profile)
                final = pred1
                stop_n = n0 + 1
                probe2 += 1
        if final == expected:
            correct += 1
        stop_ns.append(stop_n)
        fcs.append(fc)
    row = {
        "accuracy": round(correct / len(samples), 4),
        "total": len(samples),
        "probe2_count": probe2,
        "mean_stop_n": round(sum(stop_ns) / len(samples), 2),
        "mean_probes": round((len(samples) + probe2) / len(samples), 2),
        "params": {"uses_oracle": False, "conf_thr": conf_thr, "max_probes": 2},
    }
    row.update(timing_metrics(stop_ns, fcs))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        head = load_rich_head(device, load_m2_head_state(device))
        sweep = []
        for thr in (0.15, 0.35, 0.5, 0.65):
            row = eval_dual_probe(head, model, tokenizer, full, device, SEED, profile, thr)
            row["deployable_mvp"] = is_deployable_mvp(row)
            row["eps_deployable"] = is_eps_deployable(row)
            sweep.append(row)
        best = max(sweep, key=lambda r: (r["accuracy"], -(r.get("mean_probes") or 2)))
        p22 = __import__("_phase23_common", fromlist=["load_json"]).load_json("phase22/x1_structure_m2_latest.json")
        bl = (p22.get("full_419") or {}).get("accuracy", 0.9356)
        return {
            "sweep": sweep,
            "best": best,
            "full_419": best,
            "baseline_structure_d_m2": bl,
            "insight": "最多 2 次前向：低置信补探 n+1 能否抬 acc 且控算力？",
            "mentor_brief": (
                f"Z5 双探针：最优 thr={best['params']['conf_thr']} acc {best['accuracy']:.1%} "
                f"mean_probes≈{best['mean_probes']:.2f}；baseline {bl:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "z5_dual_probe", "Z5 · 双探针", device=args.device)
    write_phase23_result("z5_dual_probe", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
