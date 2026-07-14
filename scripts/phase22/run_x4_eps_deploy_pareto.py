#!/usr/bin/env python3
"""X4 · ε-deploy Pareto：knn / structure_d / 跳数分治 / 二段式 / ε-stop 全对比定稿。"""
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
    CAP, FINE_GRID, MIN_N, SEED, is_deployable_mvp, is_eps_deployable, is_feasible,
    load_full_dataset, load_json, load_m2_head_state, load_rich_head, load_splits,
    row_summary, timed_run, timing_metrics, write_phase22_result,
)
from boundary_budget import make_structure_budget_fn
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from stop_head import evaluate_rich_stop, split_train_val_samples


def _candidate_from_json(d: dict, cid: str, label: str, **defaults) -> dict:
    full = d.get("full_419") or d.get("best_acc") or d.get("best") or {}
    if isinstance(full, list):
        full = full[0] if full else {}
    row = {**defaults, **{k: full.get(k) for k in (
        "accuracy", "stop_timing_acc", "mean_stop_n", "timing_eps1", "timing_eps2", "deployable_mvp",
    )}}
    row["id"] = cid
    row["label"] = label
    row["feasible"] = is_feasible(full) if full.get("stop_timing_acc") is not None else False
    row["eps_deployable"] = is_eps_deployable(full) if full.get("timing_eps1") is not None else None
    if row.get("deployable_mvp") is None:
        row["deployable_mvp"] = is_deployable_mvp(full) if full else False
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        candidates = []

        p17 = load_json("phase17/s1_corrected_final_latest.json")
        knn = (p17.get("deployable_mvp") or {})
        candidates.append({
            "id": "knn_min3_full", "label": "knn_min3 (Phase17)",
            "accuracy": knn.get("accuracy", 0.926),
            "stop_timing_acc": knn.get("timing", 0.37),
            "mean_stop_n": knn.get("mean_n", 3.37),
            "timing_eps1": None,
            "single_forward": False,
            "deployable_mvp": True, "feasible": False, "eps_deployable": None,
        })

        w2 = load_json("phase21/w2_meta_budget_latest.json")
        candidates.append(_candidate_from_json(
            w2, "structure_d", "structure_d 单次前向 (P21)",
            single_forward=True, timing_eps1=None,
        ))

        x1 = load_json("phase22/x1_structure_m2_latest.json")
        if x1:
            candidates.append(_candidate_from_json(
                x1, "structure_d_m2", "structure_d+M2 (X1)", single_forward=False,
            ))

        x2 = load_json("phase22/x2_hop_split_budget_latest.json")
        if x2:
            best = x2.get("best_acc") or {}
            candidates.append({
                "id": f"hop_split_{best.get('strategy', 'unknown')}",
                "label": f"跳数分治 {best.get('strategy')} (X2)",
                "accuracy": best.get("accuracy"),
                "stop_timing_acc": best.get("stop_timing_acc"),
                "mean_stop_n": best.get("mean_stop_n"),
                "timing_eps1": best.get("timing_eps1"),
                "single_forward": True,
                "deployable_mvp": best.get("deployable_mvp"),
                "feasible": best.get("feasible"),
                "eps_deployable": best.get("eps_deployable"),
            })

        x3 = load_json("phase22/x3_epsilon_stop_train_latest.json")
        if x3:
            candidates.append(_candidate_from_json(
                x3, "epsilon_stop_head", "ε-stop head (X3)", single_forward=False,
            ))

        w1 = load_json("phase21/w1_epsilon_timing_latest.json")
        w1full = next((c for c in w1.get("configs", []) if c.get("split") == "full_419" and c.get("min_n") == 3), {})
        if w1full:
            candidates.append({
                "id": "m2_min3_eps_audit", "label": "M2 min3 ε审计 (P21 W1)",
                "accuracy": w1full.get("accuracy"),
                "stop_timing_acc": w1full.get("timing_eps0"),
                "timing_eps1": w1full.get("timing_eps1"),
                "timing_eps2": w1full.get("timing_eps2"),
                "mean_stop_n": w1full.get("mean_stop_n"),
                "single_forward": False,
                "deployable_mvp": w1full.get("accuracy", 0) >= 0.863,
                "feasible": w1full.get("feasible_strict"),
                "eps_deployable": w1full.get("feasible_eps1"),
            })

        # Live baseline: structure_d single forward on full (fresh ε)
        model, tokenizer, device, profile = load_model_bundle(args.device)
        full = load_full_dataset()
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        budget_fn = make_structure_budget_fn(min_n=MIN_N, cap=CAP)
        from boundary_budget import evaluate_upfront_budget_stop
        from stop_head import first_correct_step
        row = evaluate_upfront_budget_stop(
            model, tokenizer, full, budget_fn,
            strategy_id="structure_d_live", strategy_label="structure_d_live",
            cap=CAP, min_n=MIN_N, device=device, seed=SEED,
            predict_fn=lambda m, t, s, n, d, ss, ep: predict_at_n(m, t, s, n, d, seed=ss, eval_profile=ep),
            expected_fn=expected_answer, eval_profile=profile,
            extra_params={"uses_oracle": False, "single_forward": True},
        )
        stop_ns, fcs = [], []
        for idx, sample in enumerate(full):
            fc, _ = first_correct_step(
                model, tokenizer, sample, cap=CAP, device=device, seed=SEED + idx * 31,
                predict_fn=pfn, expected_fn=expected_answer, eval_profile=profile,
            )
            stop_ns.append(budget_fn(sample))
            fcs.append(fc)
        row.update(timing_metrics(stop_ns, fcs))
        live = row_summary(row, "structure_d_live", split="full_419")
        candidates.append({**live, "id": "structure_d_live", "label": "structure_d 复现 (X4)", "single_forward": True})

        best_acc = max(candidates, key=lambda c: c.get("accuracy") or 0)
        best_eps = max(candidates, key=lambda c: (c.get("timing_eps1") or 0, c.get("accuracy") or 0))
        best_strict = max(candidates, key=lambda c: (c.get("stop_timing_acc") or 0, c.get("accuracy") or 0))
        deploy_mvp = [c for c in candidates if c.get("deployable_mvp")]
        eps_dep = [c for c in candidates if c.get("eps_deployable")]

        return {
            "candidates": candidates,
            "best_acc": best_acc,
            "best_eps1": best_eps,
            "best_strict_timing": best_strict,
            "deployable_mvp_ids": [c["id"] for c in deploy_mvp],
            "eps_deployable_ids": [c["id"] for c in eps_dep],
            "feasible_any": any(c.get("feasible") for c in candidates),
            "eps_deployable_any": bool(eps_dep),
            "insight": "ε-deploy Pareto：并列推荐 acc 冠军 vs ε-timing 达标方案。",
            "mentor_brief": (
                f"X4 定稿：acc 冠军 {best_acc['id']} {best_acc.get('accuracy', 0):.1%}；"
                f"ε=1 最高 {best_eps['id']} {best_eps.get('timing_eps1', 0):.1%}；"
                f"strict 最高 {best_strict['id']} {best_strict.get('stop_timing_acc', 0):.1%}。"
            ),
            "sample_count": 419,
            "device": str(device),
        }

    path = timed_run(run_body, "x4_eps_deploy_pareto", "X4 · ε-Pareto", device=args.device)
    write_phase22_result("x4_eps_deploy_pareto", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
