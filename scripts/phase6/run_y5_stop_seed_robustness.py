#!/usr/bin/env python3
"""Y5 · soft_floor / hop_split 提示随机性鲁棒性（test 168）。"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase6_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    make_predict_fn,
    timed_run,
    write_phase6_result,
)
from evaluate_coconut import expected_answer  # noqa: E402
from graph_utils import build_eval_prompt  # noqa: E402
from stop_head_tracks import (  # noqa: E402
    evaluate_hop_split_first_correct_stop,
    evaluate_soft_floor_first_correct_stop,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3,4,42,99")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        report = {}

        for label, fn, extra in (
            ("soft_floor_fc", evaluate_soft_floor_first_correct_stop, {}),
            ("hop_split_fc", evaluate_hop_split_first_correct_stop, {"split_depth": 4}),
        ):
            accs = []
            timings = []
            for seed in seeds:
                kw = dict(
                    cap=args.cap,
                    min_n=args.min_n,
                    device=device,
                    seed=seed,
                    predict_fn=predict_fn,
                    expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt,
                    eval_profile=profile,
                    **extra,
                )
                row = fn(model, tokenizer, test_set, **kw)
                accs.append(row["accuracy"])
                timings.append(row.get("stop_timing_acc") or 0.0)
            report[label] = {
                "seeds": seeds,
                "accuracies": accs,
                "mean_acc": round(statistics.mean(accs), 4),
                "stdev_acc": round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0,
                "min_acc": round(min(accs), 4),
                "max_acc": round(max(accs), 4),
                "mean_timing": round(statistics.mean(timings), 4),
            }

        return {
            "robustness": report,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "min_n": args.min_n,
            "device": str(device),
            "insight": "X5 仅覆盖 fixed_3/auto_route；本实验补 winners 的 seed 方差。",
        }

    path = timed_run(run_body, "y5_stop_seed_robustness", "Y5 · stop 策略 seed 鲁棒性", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase6_result("y5_stop_seed_robustness", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
