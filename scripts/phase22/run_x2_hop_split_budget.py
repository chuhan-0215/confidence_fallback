#!/usr/bin/env python3
"""X2 · 跳数分治元预算：3跳 structure_d / 4跳 d-1·prompt·lookup·knn。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase22_common import (
    CAP, MIN_N, SEED, load_full_dataset, load_splits, row_summary, timed_run, write_phase22_result,
)
from boundary_budget import (
    PromptD4BinaryMLP,
    blind_depth,
    build_prompt_budget_labels,
    evaluate_upfront_budget_stop,
    make_d4_knn_budget_fn,
    make_d_minus_one_budget_fn,
    make_lookup_budget_fn,
    make_prompt_d4_budget_fn,
    make_structure_budget_fn,
    train_d4_knn_bank,
    train_lookup_budget_table,
)
from evaluate_coconut import expected_answer
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import first_correct_step, split_train_val_samples


def make_hop_split_budget_fn(four_hop_fn, *, min_n: int, cap: int):
    def _fn(sample: dict) -> int:
        d = blind_depth(sample)
        if d < 4:
            return max(min_n, min(cap, d))
        return four_hop_fn(sample)
    return _fn


def _load_prompt_head(device):
    for rel in (
        "phase21/w2_prompt_budget_mlp.pt",
        "results/phase21/w2_prompt_budget_mlp.pt",
    ):
        for base in (ROOT / "outbox/results/from_a800", ROOT / "results"):
            p = base / rel
            if p.is_file():
                ckpt = torch.load(p, map_location=device, weights_only=False)
                in_dim = ckpt.get("meta", {}).get("in_dim")
                if in_dim is None:
                    sd = ckpt["state_dict"]
                    in_dim = sd["net.0.weight"].shape[1]
                head = PromptD4BinaryMLP(in_dim=in_dim).to(device)
                head.load_state_dict(ckpt["state_dict"])
                return head
    return None


@torch.no_grad()
def evaluate_budget_with_eps(model, tokenizer, samples, budget_fn, *, cap, device, seed, pfn, profile):
    row = evaluate_upfront_budget_stop(
        model, tokenizer, samples, budget_fn,
        strategy_id="hop_split", strategy_label="hop_split",
        cap=cap, min_n=MIN_N, device=device, seed=seed,
        predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
        extra_params={"uses_oracle": False, "single_forward": True},
    )
    stop_ns, fcs = [], []
    for idx, sample in enumerate(samples):
        n_pred = budget_fn(sample)
        fc, _ = first_correct_step(
            model, tokenizer, sample, cap=cap, device=device, seed=seed + idx * 31,
            predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
            expected_fn=expected_answer, eval_profile=profile,
        )
        stop_ns.append(n_pred)
        fcs.append(fc)
    from _phase22_common import timing_metrics
    row.update(timing_metrics(stop_ns, fcs))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        train_set, test_set = load_splits()
        full = load_full_dataset()
        train_sub, val_sub = split_train_val_samples(train_set, val_ratio=0.2, seed=43)
        pfn = lambda m, t, s, n, d, ss, ep: predict_at_n(m, t, s, n, d, seed=ss, eval_profile=ep)

        train_rows = build_prompt_budget_labels(
            model, tokenizer, train_sub, cap=CAP, min_n=MIN_N, device=device, seed=42,
            predict_fn=lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile),
            expected_fn=expected_answer, eval_profile=profile,
        )
        lookup_table, lookup_meta = train_lookup_budget_table(train_rows, min_n=MIN_N, cap=CAP)
        knn_bank, knn_meta = train_d4_knn_bank(train_rows, feature_key="joint_features")
        prompt_head = _load_prompt_head(device)

        three_hop = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
        variants = [
            ("d_minus_one", make_hop_split_budget_fn(
                make_d_minus_one_budget_fn(min_n=MIN_N, cap=CAP), min_n=MIN_N, cap=CAP)),
            ("lookup_table", make_hop_split_budget_fn(
                make_lookup_budget_fn(lookup_table, min_n=MIN_N, cap=CAP), min_n=MIN_N, cap=CAP)),
            ("knn_joint", make_hop_split_budget_fn(
                make_d4_knn_budget_fn(knn_bank, k=5, min_n=MIN_N, cap=CAP), min_n=MIN_N, cap=CAP)),
        ]
        if prompt_head is not None:
            variants.append(("prompt_joint", make_hop_split_budget_fn(
                make_prompt_d4_budget_fn(
                    prompt_head, model, tokenizer, min_n=MIN_N, cap=CAP,
                    device=device, eval_profile=profile, seed_base=SEED,
                ), min_n=MIN_N, cap=CAP)))

        results = []
        for sid, budget_fn in variants:
            for split_name, samples in (("test", test_set), ("full_419", full)):
                row = evaluate_budget_with_eps(
                    model, tokenizer, samples, budget_fn,
                    cap=CAP, device=device, seed=SEED, pfn=pfn, profile=profile,
                )
                row["params"]["strategy"] = sid
                row["params"]["three_hop_rule"] = "structure_d"
                results.append(row_summary(row, sid, split=split_name))

        full_rows = [r for r in results if r["split"] == "full_419"]
        best_acc = max(full_rows, key=lambda r: r["accuracy"] or 0)
        best_eps = max(full_rows, key=lambda r: (r.get("timing_eps1") or 0, r.get("accuracy") or 0))
        baseline = evaluate_budget_with_eps(
            model, tokenizer, full, three_hop, cap=CAP, device=device, seed=SEED, pfn=pfn, profile=profile,
        )
        baseline["params"]["strategy"] = "structure_d_only"
        bl = row_summary(baseline, "structure_d_only", split="full_419")

        return {
            "results": results,
            "lookup_meta": lookup_meta,
            "knn_meta": knn_meta,
            "baseline_structure_d": bl,
            "best_acc": best_acc,
            "best_eps1": best_eps,
            "full_419": best_acc,
            "deployable_mvp": any(r.get("deployable_mvp") for r in full_rows),
            "eps_deployable_any": any(r.get("eps_deployable") for r in full_rows),
            "insight": "3跳 structure_d + 4跳专用预算：能否 beat 纯 structure_d 93.6% 或抬 ε-timing？",
            "mentor_brief": (
                f"X2 跳数分治：最优 acc {best_acc['strategy']} {best_acc['accuracy']:.1%}；"
                f"最优 ε=1 {best_eps['strategy']} {best_eps.get('timing_eps1', 0):.1%}；"
                f"baseline structure_d {bl['accuracy']:.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "x2_hop_split_budget", "X2 · 跳数分治", device=args.device)
    write_phase22_result("x2_hop_split_budget", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
