#!/usr/bin/env python3
"""K4 · 选最优 checkpoint 全量 419 验证。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase12_common import CAP, MIN_N, SEED, is_feasible, load_rich_head, timed_run, write_phase12_result
from evaluate_coconut import expected_answer, load_dataset
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop

K2_JSON = ROOT / "results" / "phase12" / "k2_stable_correct_train_latest.json"
K3_JSON = ROOT / "results" / "phase12" / "k3_long_head_train_latest.json"
K2_CKPT = ROOT / "results" / "phase12" / "k2_stable_correct_head.pt"
K3_CKPT = ROOT / "results" / "phase12" / "k3_long_head.pt"


def _pick_best():
    candidates = []
    for jpath, cpath in ((K2_JSON, K2_CKPT), (K3_JSON, K3_CKPT)):
        if jpath.is_file() and cpath.is_file():
            d = json.loads(jpath.read_text(encoding="utf-8"))
            candidates.append((d.get("feasible", False), d.get("test", {}).get("accuracy", 0),
                               d.get("threshold", 0.5), cpath, jpath.stem))
    if not candidates:
        return 0.5, ROOT / "results" / "phase10" / "m2_enough_stop_head.pt", "m2_fallback"
    candidates.sort(reverse=True)
    _, _, thr, ckpt, name = candidates[0]
    return thr, ckpt, name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        thr, ckpt_path, source = _pick_best()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head = load_rich_head(device, ckpt.get("state_dict"))

        full_set = load_dataset(ROOT / "data" / "prosqa_test_graph_4_coconut.json", None)
        policies = make_policies(cap=CAP)
        fixed3 = evaluate_policy(model, tokenizer, full_set, policies["fixed_3"], device, cap=CAP, eval_profile=profile)
        row = evaluate_rich_stop(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )
        return {
            "source": source,
            "threshold": thr,
            "test": row,
            "feasible": is_feasible(row),
            "fixed_3_full_acc": fixed3["accuracy"],
            "insight": "全量 419 验证最优 K2/K3 checkpoint。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "k4_full419", "K4 · 全量验证", device=args.device)
    import json as _json
    write_phase12_result("k4_full419", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
