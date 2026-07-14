#!/usr/bin/env python3
"""Z3 · Y8 hybrid 精确 probe 计数（逐题 trace）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hybrid_eval import evaluate_hybrid, evaluate_two_probe_hybrid
from _phase7_common import load_model_bundle, load_test_split, timed_run, write_phase7_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        estimated = evaluate_two_probe_hybrid(
            model, tokenizer, test_set,
            cap=args.cap, min_n=args.min_n, device=device, seed=args.seed, profile=profile,
        )
        exact = evaluate_hybrid(
            model, tokenizer, test_set,
            cap=args.cap, min_n=args.min_n, device=device, seed=args.seed, profile=profile,
            count_probes_exact=True,
        )
        exact["strategy"] = "two_probe_exact_probe_count"
        delta = round(exact["mean_forward_probes"] - estimated["mean_forward_probes"], 3)
        return {
            "audit": {
                "estimated": estimated,
                "exact": exact,
                "probe_delta": delta,
            },
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "seed": args.seed,
            "insight": "精确计数每次 predict_at_n；与 Y8 估计 probes 对比。",
            "device": str(device),
        }

    path = timed_run(run_body, "z3_y8_probe_audit", "Z3 · Y8 probe 审计", device=args.device)
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase7_result("z3_y8_probe_audit", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
