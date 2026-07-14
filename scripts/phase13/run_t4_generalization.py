#!/usr/bin/env python3
"""T4 · 全量 419 + seed77，选 T1/T2/T3 最优。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase13_common import CAP, MIN_N, SEED, is_feasible, load_full_dataset, load_m2_head_state, load_rich_head, load_splits, timed_run, write_phase13_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop
from stop_head_tracks import evaluate_rich_or_stable_stop

CANDIDATES = [
    ("t1", ROOT / "results/phase13/t1_head_or_stable_latest.json", None, "or_stable"),
    ("t2", ROOT / "results/phase13/t2_first_correct_train_latest.json", ROOT / "results/phase13/t2_first_correct_head.pt", "auto"),
    ("t3", ROOT / "results/phase13/t3_earliest_stop_train_latest.json", ROOT / "results/phase13/t3_earliest_stop_head.pt", "correctness"),
]


def _eval_fn(mode, head, model, tokenizer, samples, thr, patience, device, profile, pfn):
    if mode == "or_stable":
        return evaluate_rich_or_stable_stop(
            head, model, tokenizer, samples, cap=CAP, min_n=MIN_N, threshold=thr, patience=patience,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
    return evaluate_rich_stop(
        head, model, tokenizer, samples, cap=CAP, min_n=MIN_N, threshold=thr,
        device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
        build_prompt_fn=build_eval_prompt, eval_profile=profile,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)

        best = None
        best_score = (-1, -1.0, -1.0)
        for name, jpath, ckpt, mode in CANDIDATES:
            if not jpath.is_file():
                continue
            data = json.loads(jpath.read_text(encoding="utf-8"))
            test = data.get("test") or data.get("head_or_stable_test") or {}
            score = (1 if data.get("feasible") else 0, test.get("accuracy", 0), test.get("stop_timing_acc", 0))
            if score > best_score:
                best_score = score
                if name == "t1":
                    pick = data.get("picked") or {}
                    thr, patience, eval_mode = pick.get("threshold", 0.5), pick.get("patience", 2), "or_stable"
                elif name == "t2":
                    hv = data.get("head_or_stable_test")
                    if data.get("feasible") and hv:
                        thr, patience, eval_mode = data["variants"][data["best_variant"]]["threshold"], 2, "or_stable"
                    else:
                        thr = data["variants"][data["best_variant"]]["threshold"]
                        patience, eval_mode = 2, "correctness"
                else:
                    thr, patience, eval_mode = data.get("threshold", 0.5), 2, "correctness"
                best = (name, ckpt, thr, patience, eval_mode)

        if best is None:
            raise FileNotFoundError("需要 T1/T2/T3 结果")
        name, ckpt, thr, patience, eval_mode = best
        if ckpt and ckpt.is_file():
            state = torch.load(ckpt, map_location=device, weights_only=False).get("state_dict")
            head = load_rich_head(device, state)
        else:
            head = load_rich_head(device, load_m2_head_state(device))

        full_set = load_full_dataset()
        ar_full = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        full_row = _eval_fn(eval_mode, head, model, tokenizer, full_set, thr, patience, device, profile, pfn)

        _, alt_test = load_splits(train_ratio=0.6, seed=77)
        ar_alt = evaluate_policy(model, tokenizer, alt_test, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        alt_row = _eval_fn(eval_mode, head, model, tokenizer, alt_test, thr, patience, device, profile, pfn)

        return {
            "source": name,
            "threshold": thr,
            "patience": patience,
            "eval_mode": eval_mode,
            "full_419": {"test": full_row, "auto_route_acc": ar_full["accuracy"], "feasible": is_feasible(full_row)},
            "alt_split_seed77": {"test": alt_row, "auto_route_acc": ar_alt["accuracy"], "feasible": is_feasible(alt_row)},
            "feasible": is_feasible(full_row) and is_feasible(alt_row),
            "fully_proven": is_feasible(full_row) and is_feasible(alt_row),
            "insight": "泛化验证最优 T1/T2/T3。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "t4_generalization", "T4 · 泛化", device=args.device)
    import json as _json
    write_phase13_result("t4_generalization", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
