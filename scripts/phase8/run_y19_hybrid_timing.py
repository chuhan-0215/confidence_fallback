#!/usr/bin/env python3
"""Y19 · hybrid timing 指标（one_shot@fc 与 retry@fc）。"""
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
    ap.add_argument("--min-n", type=int, default=DEPLOY_DEFAULTS["min_n"])
    ap.add_argument("--seed", type=int, default=DEPLOY_DEFAULTS["seed"])
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        row = evaluate_hybrid_traced(
            model, tokenizer, test_set,
            cap=args.cap, min_n=args.min_n, device=device, seed=args.seed, profile=profile,
        )
        return {
            "timing": {
                "stop_timing_acc": row["stop_timing_acc"],
                "one_probe_success_rate": row["one_probe_success_rate"],
                "accuracy": row["accuracy"],
                "mean_forward_probes": row["mean_forward_probes"],
                "by_path": row["by_path"],
            },
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "deploy": DEPLOY_DEFAULTS,
            "device": str(device),
            "insight": "补 Y14 hybrid timing 缺失；one_shot 计 fc==n0，retry 计 stop_n==fc。",
        }

    path = timed_run(run_body, "y19_hybrid_timing", "Y19 · hybrid timing", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase8_result("y19_hybrid_timing", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
