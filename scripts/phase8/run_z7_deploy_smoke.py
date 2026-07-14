#!/usr/bin/env python3
"""Z7 · 部署配方锁定 smoke（canonical JSON 输出）。"""
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
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        row = evaluate_hybrid_traced(
            model,
            tokenizer,
            test_set,
            cap=DEPLOY_DEFAULTS["cap"],
            min_n=DEPLOY_DEFAULTS["min_n"],
            device=device,
            seed=DEPLOY_DEFAULTS["seed"],
            profile=profile,
        )
        return {
            "deploy_recipe": DEPLOY_DEFAULTS,
            "eval_split": "test_40pct",
            "sample_count": len(test_set),
            "metrics": {
                "accuracy": row["accuracy"],
                "correct": row["correct"],
                "total": row["total"],
                "mean_forward_probes": row["mean_forward_probes"],
                "one_probe_success_rate": row["one_probe_success_rate"],
                "stop_timing_acc": row["stop_timing_acc"],
            },
            "acceptance": {
                "min_accuracy": 0.95,
                "max_mean_probes": 1.5,
                "pass_accuracy": row["accuracy"] >= 0.95,
                "pass_probes": row["mean_forward_probes"] <= 1.5,
            },
            "device": str(device),
            "insight": "CI/回归用 canonical smoke；acc≥95% 且 probes≤1.5 为 pass。",
        }

    path = timed_run(run_body, "z7_deploy_smoke", "Z7 · 部署 smoke", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase8_result("z7_deploy_smoke", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
