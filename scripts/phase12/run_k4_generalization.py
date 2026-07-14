#!/usr/bin/env python3
"""K4 · 全量 419 + seed77，用 K1/K2/K3 最优 checkpoint。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import CAP, MIN_N, SEED, is_feasible, load_full_dataset, load_rich_head, load_splits, timed_run, write_phase12_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop

CANDIDATES = [
    ("k3_hybrid", ROOT / "results" / "phase12" / "k3_hybrid_distill_latest.json", ROOT / "results" / "phase12" / "k3_hybrid_distill.pt"),
    ("k2_stable", ROOT / "results" / "phase12" / "k2_stable_correct_train_latest.json", ROOT / "results" / "phase12" / "k2_stable_correct_head.pt"),
    ("k3_long", ROOT / "results" / "phase12" / "k3_long_head_train_latest.json", ROOT / "results" / "phase12" / "k3_long_head.pt"),
    ("k1_m2", ROOT / "results" / "phase12" / "k1_threshold_sweep_latest.json", None),
]


def _best_from_json(path: Path) -> tuple[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("feasible") and data.get("threshold") is not None:
        return "feasible", data["threshold"]
    if "best_timing_threshold" in data:
        bt = data["best_timing_threshold"]
        return "best_timing", bt["threshold"]
    if "modes" in data:
        best = data.get("best_mode")
        return best, data["modes"][best]["threshold"]
    if "variants" in data:
        best = data.get("best_variant")
        return best, data["variants"][best]["threshold"]
    return data.get("test", {}).get("strategy", "learned"), data.get("threshold", 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)

        picked = None
        best_score = (-1, -1.0, -1.0)
        for name, jpath, ckpt in CANDIDATES:
            if not jpath.is_file():
                continue
            data = json.loads(jpath.read_text(encoding="utf-8"))
            test = data.get("test")
            if not test and data.get("variants"):
                test = data["variants"][data["best_variant"]]["test"]
            if not test and data.get("best_timing_threshold"):
                test = data["best_timing_threshold"]
            thr = data.get("threshold") or (test or {}).get("threshold") or _best_from_json(jpath)[1]
            score = (
                1 if data.get("feasible") else 0,
                (test or {}).get("accuracy", 0),
                (test or {}).get("stop_timing_acc", 0),
            )
            if score > best_score:
                best_score = score
                picked = (name, jpath, ckpt, thr)

        if picked is None:
            raise FileNotFoundError("需要 K1/K2/K3 结果")
        name, jpath, ckpt, thr = picked
        if ckpt and ckpt.is_file():
            state = torch.load(ckpt, map_location=device, weights_only=False).get("state_dict")
            head = load_rich_head(device, state)
        elif name == "k1_m2":
            from _phase12_common import load_m2_head_state
            data = json.loads(jpath.read_text(encoding="utf-8"))
            bt = data.get("best_timing_threshold") or {}
            thr = bt.get("threshold", thr)
            head = load_rich_head(device, load_m2_head_state(device))
        else:
            from _phase12_common import load_m2_head_state
            head = load_rich_head(device, load_m2_head_state(device))

        full_set = load_full_dataset()
        ar_full = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        full_row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        _, alt_test = load_splits(train_ratio=0.6, seed=77)
        ar_alt = evaluate_policy(model, tokenizer, alt_test, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        alt_row = evaluate_rich_stop(
            head, model, tokenizer, alt_test, cap=CAP, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED + 7, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )

        full_ok = is_feasible(full_row, ar_full["accuracy"])
        alt_ok = is_feasible(alt_row, ar_alt["accuracy"])
        return {
            "source": name,
            "threshold": thr,
            "full_419": {"test": full_row, "auto_route_acc": ar_full["accuracy"], "feasible": full_ok},
            "alt_split_seed77": {"test": alt_row, "auto_route_acc": ar_alt["accuracy"], "feasible": alt_ok},
            "feasible": full_ok and alt_ok,
            "fully_proven": full_ok and alt_ok,
            "insight": "不用 streak；选 K1/K2/K3 最优 head。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "k4_generalization", "K4 · 泛化", device=args.device)
    import json as _json
    write_phase12_result("k4_generalization", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
