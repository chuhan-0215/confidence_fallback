#!/usr/bin/env python3
"""Y18 · hybrid 跨子集泛化（core + boundary_push）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase8_common import DEPLOY_DEFAULTS, evaluate_hybrid, load_model_bundle, timed_run, write_phase8_result
from dataset_registry import load_slice
from phase5.run_y2_hop_split_slices import SLICE_GROUPS


def eval_slice_hybrid(model, tokenizer, device, profile, slice_id, max_samples, cap, min_n, seed):
    meta, samples = load_slice(slice_id, max_samples=max_samples)
    row = evaluate_hybrid(
        model, tokenizer, samples, cap=cap, min_n=min_n, device=device, seed=seed, profile=profile,
    )
    return {
        "slice_id": slice_id,
        "label": meta.get("label"),
        "count": len(samples),
        "hybrid": {
            "accuracy": row["accuracy"],
            "mean_forward_probes": row["mean_forward_probes"],
            "one_probe_success_rate": row["one_probe_success_rate"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group", choices=list(SLICE_GROUPS), default="core")
    ap.add_argument("--cap", type=int, default=DEPLOY_DEFAULTS["cap"])
    ap.add_argument("--min-n", type=int, default=DEPLOY_DEFAULTS["min_n"])
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=DEPLOY_DEFAULTS["seed"])
    args = ap.parse_args()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    rows = [
        eval_slice_hybrid(
            model, tokenizer, device, profile, sid, args.max_samples, args.cap, args.min_n, args.seed
        )
        for sid in SLICE_GROUPS[args.group]
    ]
    exp_id = f"y18_hybrid_slices_{args.group}"

    path = timed_run(
        lambda: {"slice_group": args.group, "slices": rows, "seed": args.seed, "device": str(device)},
        exp_id,
        f"Y18 · hybrid 跨子集 ({args.group})",
        device=args.device,
    )
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    write_phase8_result(exp_id, data)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
