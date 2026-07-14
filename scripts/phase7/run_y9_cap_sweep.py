#!/usr/bin/env python3
"""Y9 · cap sweep：soft_floor vs two_probe × cap∈{6,8,10}。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hybrid_eval import evaluate_hybrid
from _phase7_common import load_model_bundle, load_test_split, timed_run, write_phase7_result
from evaluate_coconut import expected_answer
from graph_utils import build_eval_prompt
from phase5._phase5_common import make_predict_fn
from stop_head_tracks import evaluate_soft_floor_first_correct_stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--caps", default="6,8,10")
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    caps = [int(x) for x in args.caps.split(",") if x.strip()]

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        rows = []
        for cap in caps:
            sf = evaluate_soft_floor_first_correct_stop(
                model,
                tokenizer,
                test_set,
                cap=cap,
                min_n=args.min_n,
                device=device,
                seed=args.seed,
                predict_fn=predict_fn,
                expected_fn=expected_answer,
                build_prompt_fn=build_eval_prompt,
                eval_profile=profile,
            )
            rows.append(
                {
                    "strategy": "soft_floor_fc",
                    "cap": cap,
                    "accuracy": sf["accuracy"],
                    "stop_timing_acc": sf.get("stop_timing_acc"),
                    "mean_stop_n": sf.get("mean_stop_n"),
                }
            )
            hy = evaluate_hybrid(
                model,
                tokenizer,
                test_set,
                cap=cap,
                min_n=args.min_n,
                device=device,
                seed=args.seed,
                profile=profile,
            )
            rows.append({"cap": cap, **hy})
        return {
            "sweep": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "seed": args.seed,
            "min_n": args.min_n,
            "device": str(device),
            "insight": "cap 对 ext 链影响；生产 4 跳默认 cap=8 是否足够。",
        }

    path = timed_run(run_body, "y9_cap_sweep", "Y9 · cap sweep", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("y9_cap_sweep", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
