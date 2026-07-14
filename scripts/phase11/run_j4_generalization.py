#!/usr/bin/env python3
"""J4 · 泛化：全量 419 + 换 seed 切分，验证不是过拟合单一 split。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase11_common import CAP, MIN_N, SEED, is_feasible, load_full_dataset, load_rich_head, proof_status, timed_run, write_phase11_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase4._phase4_common import load_model_bundle
from run_adaptive_stop_experiment import predict_at_n
from run_auto_submit_experiment import evaluate_policy, make_policies
from stop_head import evaluate_rich_stop, evaluate_streak_gated_stop, split_train_val_samples

J2_CKPT = ROOT / "results" / "phase11" / "j2_joint_warmstart.pt"
J2_JSON = ROOT / "results" / "phase11" / "j2_joint_warmstart_latest.json"
J3_JSON = ROOT / "results" / "phase11" / "j3_joint_deep_latest.json"


def _pick_best_config() -> tuple[float, str, bool]:
    candidates = []
    for path, use_streak in ((J2_JSON, True), (J3_JSON, False)):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if path == J2_JSON:
            best = data.get("best_variant")
            if best and data.get("variants"):
                row = data["variants"][best]
                candidates.append((row.get("threshold", 0.5), best, "streak" in best))
        else:
            candidates.append((data.get("threshold", 0.5), "j3", data.get("streak_test") and data.get("feasible")))
    if not candidates:
        return 0.5, "default", False
    # prefer j2/j3 json feasible flag
    for thr, name, streak in candidates:
        src = J2_JSON if "j2" in name or name in ("correctness_feasible", "streak_feasible", "correctness_balanced") else J3_JSON
        if src.is_file():
            d = json.loads(src.read_text(encoding="utf-8"))
            if d.get("feasible"):
                return thr, name, streak
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        pfn = lambda s, n, ss: predict_at_n(model, tokenizer, s, n, device, seed=ss, eval_profile=profile)
        policies = make_policies(cap=CAP)
        thr, variant, use_streak = _pick_best_config()

        if J2_CKPT.is_file():
            ckpt = torch.load(J2_CKPT, map_location=device, weights_only=False)
            head = load_rich_head(device, ckpt.get("head_state"))
            if ckpt.get("model_state"):
                model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            raise FileNotFoundError("需要 J2 checkpoint；请先跑 j2_joint_warmstart")

        full_set = load_full_dataset()
        ar_full = evaluate_policy(model, tokenizer, full_set, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        eval_fn = evaluate_streak_gated_stop if use_streak else evaluate_rich_stop
        full_row = eval_fn(
            head, model, tokenizer, full_set, cap=CAP, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )

        from _phase11_common import load_splits
        alt_train, alt_test = load_splits(train_ratio=0.6, seed=77)
        ar_alt = evaluate_policy(model, tokenizer, alt_test, policies["auto_route"], device, cap=CAP, eval_profile=profile)
        alt_row = eval_fn(
            head, model, tokenizer, alt_test, cap=CAP, min_n=MIN_N, threshold=thr,
            device=device, seed=SEED + 7, predict_fn=pfn, expected_fn=expected_answer,
            build_prompt_fn=build_eval_prompt, eval_profile=profile,
        )

        full_ok = is_feasible(full_row, ar_full["accuracy"])
        alt_ok = is_feasible(alt_row, ar_alt["accuracy"])
        useful = full_row.get("mean_stop_n", CAP) < CAP - 0.5 and full_row["accuracy"] >= ar_full["accuracy"] - 0.01
        return {
            "variant": variant,
            "threshold": thr,
            "use_streak": use_streak,
            "full_419": {"test": full_row, "auto_route_acc": ar_full["accuracy"], "feasible": full_ok},
            "alt_split_seed77": {"test": alt_row, "auto_route_acc": ar_alt["accuracy"], "feasible": alt_ok},
            "feasible": full_ok and alt_ok,
            "proof": proof_status(
                necessity=True,
                learnable=True,
                deploy_feasible=full_ok,
                useful=useful,
                general=alt_ok,
            ),
            "insight": "全量 + 换 seed 切分；两条都 feasible 才算「全部证明」。",
            "sample_count": len(full_set),
            "device": str(device),
        }

    path = timed_run(run_body, "j4_generalization", "J4 · 泛化验证", device=args.device)
    import json as _json
    write_phase11_result("j4_generalization", _json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
