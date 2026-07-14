#!/usr/bin/env python3
"""Y20 · hybrid min_n 扫描（test 168）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hybrid_traced import evaluate_hybrid_traced
from _phase8_common import DEPLOY_DEFAULTS, load_model_bundle, load_test_split, timed_run, write_phase8_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=DEPLOY_DEFAULTS["cap"])
    ap.add_argument("--seed", type=int, default=DEPLOY_DEFAULTS["seed"])
    ap.add_argument("--min-ns", default="1,2,3")
    args = ap.parse_args()
    min_ns = [int(x) for x in args.min_ns.split(",") if x.strip()]

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        rows = []
        for min_n in min_ns:
            row = evaluate_hybrid_traced(
                model, tokenizer, test_set,
                cap=args.cap, min_n=min_n, device=device, seed=args.seed, profile=profile,
            )
            rows.append(
                {
                    "min_n": min_n,
                    "accuracy": row["accuracy"],
                    "mean_forward_probes": row["mean_forward_probes"],
                    "stop_timing_acc": row["stop_timing_acc"],
                    "wrong_count": row["wrong_count"],
                }
            )
        return {
            "sweep": rows,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "seed": args.seed,
            "device": str(device),
            "insight": "hybrid 是否可从 min_n=2 再挤出 acc（对照 Y3/Z2）。",
        }

    path = timed_run(run_body, "y20_hybrid_min_n_sweep", "Y20 · hybrid min_n", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase8_result("y20_hybrid_min_n_sweep", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
