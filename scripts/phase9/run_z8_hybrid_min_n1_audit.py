#!/usr/bin/env python3
"""Z8 · hybrid min_n=1 审计（对照 Y20 98.21%）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase9_common import DEPLOY_DEFAULTS, load_model_bundle, load_test_split, timed_run, write_phase9_result
from phase8._hybrid_traced import evaluate_hybrid_traced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=DEPLOY_DEFAULTS["cap"])
    ap.add_argument("--seed", type=int, default=DEPLOY_DEFAULTS["seed"])
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        baseline = evaluate_hybrid_traced(
            model, tokenizer, test_set, cap=args.cap, min_n=2,
            device=device, seed=args.seed, profile=profile,
        )
        min1 = evaluate_hybrid_traced(
            model, tokenizer, test_set, cap=args.cap, min_n=1,
            device=device, seed=args.seed, profile=profile,
        )
        suspicious = []
        for f in baseline["failures"]:
            if f.get("first_correct") == 1 and f.get("blind_depth", 0) >= 3:
                suspicious.append({"idx": f["idx"], "reason": "fc1_early_on_deep", "d": f["blind_depth"]})
        return {
            "min_n_2": {
                "accuracy": baseline["accuracy"],
                "wrong_count": baseline["wrong_count"],
                "mean_forward_probes": baseline["mean_forward_probes"],
            },
            "min_n_1": {
                "accuracy": min1["accuracy"],
                "wrong_count": min1["wrong_count"],
                "mean_forward_probes": min1["mean_forward_probes"],
                "failures": min1["failures"],
            },
            "delta_acc_pp": round((min1["accuracy"] - baseline["accuracy"]) * 100, 2),
            "suspicious_fc1_deep": suspicious,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "deploy": DEPLOY_DEFAULTS,
            "device": str(device),
            "insight": "min_n=1 多对的题是否 fc=1 幻觉；若 suspicious 多则保持 min_n=2。",
        }

    path = timed_run(run_body, "z8_hybrid_min_n1_audit", "Z8 · hybrid min_n=1 审计", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase9_result("z8_hybrid_min_n1_audit", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
