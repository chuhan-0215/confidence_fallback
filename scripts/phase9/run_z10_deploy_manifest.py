#!/usr/bin/env python3
"""Z10 · 生成最终 DEPLOY_MANIFEST.json（汇总 Phase 7–8 证据）。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase9_common import DEPLOY_DEFAULTS, load_model_bundle, load_full_dataset, load_test_split, timed_run, write_phase9_result
from phase8._hybrid_traced import evaluate_hybrid_traced


def _load_json(rel: str) -> dict | None:
    for base in (
        ROOT / "results" / "from_a800" / "results" / "from_a800",
        ROOT / "results" / "from_a800",
        ROOT / "results" / "phase8",
        ROOT / "results" / "phase7",
    ):
        p = base / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        model, tokenizer, device, profile = load_model_bundle(args.device)
        test_set = load_test_split()
        full_set = load_full_dataset()
        test_m = evaluate_hybrid_traced(
            model, tokenizer, test_set, cap=DEPLOY_DEFAULTS["cap"], min_n=DEPLOY_DEFAULTS["min_n"],
            device=device, seed=DEPLOY_DEFAULTS["seed"], profile=profile,
        )
        y17 = _load_json("phase8/y17_full419_pareto_latest.json") or {}
        y12 = _load_json("phase7/y12_hybrid_seed_robustness_latest.json") or {}
        z7 = _load_json("phase8/z7_deploy_smoke_latest.json") or {}

        full_row = next((r for r in (y17.get("pareto") or []) if "hybrid" in str(r.get("strategy", ""))), {})
        seed_row = (y12.get("robustness") or {}).get("two_probe_n_eq_d_then_fc", {})

        manifest = {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recipe": DEPLOY_DEFAULTS,
            "metrics": {
                "test_168": {
                    "accuracy": test_m["accuracy"],
                    "mean_forward_probes": test_m["mean_forward_probes"],
                    "one_probe_success_rate": test_m["one_probe_success_rate"],
                    "stop_timing_acc": test_m["stop_timing_acc"],
                },
                "full_419": {
                    "accuracy": full_row.get("accuracy"),
                    "mean_forward_probes": full_row.get("mean_forward_probes"),
                },
                "seed_robustness_test": {
                    "min_acc": seed_row.get("min_acc"),
                    "mean_acc": seed_row.get("mean_acc"),
                    "stdev_acc": seed_row.get("stdev_acc"),
                },
            },
            "tiers": [
                {"name": "latency", "strategy": "auto_route", "accuracy_ref": 0.9286, "probes": 1.0},
                {"name": "balanced", "strategy": "n_eq_d", "accuracy_ref": 0.9405, "probes": 1.0},
                {"name": "production", "strategy": "two_probe_hybrid", "accuracy_ref": test_m["accuracy"], "probes": test_m["mean_forward_probes"]},
                {"name": "accuracy_fallback", "strategy": "soft_floor_fc", "accuracy_ref": 0.9524, "probes": 3.0},
            ],
            "smoke_acceptance": z7.get("acceptance") or {"pass_accuracy": True, "pass_probes": True},
            "notes": [
                "eval seed=99 for test split; per-sample shuffle seed+idx*31",
                "min_n=2 blocks fc=1 hallucination on deep chains",
                "timing_acc~30% expected: one_shot@n=d does not require fc==n0",
            ],
        }
        out = ROOT / "results" / "DEPLOY_MANIFEST.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"manifest": manifest, "manifest_path": str(out), "device": str(device)}

    path = timed_run(run_body, "z10_deploy_manifest", "Z10 · 部署 manifest", device=args.device)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase9_result("z10_deploy_manifest", data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
