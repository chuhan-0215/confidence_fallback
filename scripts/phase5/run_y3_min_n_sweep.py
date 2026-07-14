#!/usr/bin/env python3
"""Y3 · soft_floor / hop_split 的 min_n 扫描（test 40%）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase5_common import (  # noqa: E402
    load_model_bundle,
    load_test_split,
    make_predict_fn,
    timed_run,
    write_phase5_result,
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
    ap.add_argument("--min-ns", default="1,2,3,4")
    args = ap.parse_args()
    min_ns = [int(x) for x in args.min_ns.split(",") if x.strip()]

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        predict_fn = make_predict_fn(model, tokenizer, device, profile)
        rows = []
        for min_n in min_ns:
            for label, fn, extra in (
                ("soft_floor_fc", evaluate_soft_floor_first_correct_stop, {}),
                ("hop_split_fc", evaluate_hop_split_first_correct_stop, {"split_depth": 4}),
            ):
                kw = dict(
                    cap=args.cap,
                    min_n=min_n,
                    device=device,
                    seed=42,
                    predict_fn=predict_fn,
                    expected_fn=expected_answer,
                    build_prompt_fn=build_eval_prompt,
                    eval_profile=profile,
                    **extra,
                )
                row = fn(model, tokenizer, test_set, **kw)
                rows.append(
                    {
                        "strategy": label,
                        "min_n": min_n,
                        "accuracy": row["accuracy"],
                        "stop_timing_acc": row.get("stop_timing_acc"),
                        "mean_stop_n": row.get("mean_stop_n"),
                    }
                )
        return {
            "sweep": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "device": str(device),
            "insight": "min_n↑ 通常 timing↑；找 acc≥fixed_3 且 timing 最高的 deployable 点",
        }

    path = timed_run(run_body, "y3_min_n_sweep", "Y3 · min_n 扫描", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase5_result("y3_min_n_sweep", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
