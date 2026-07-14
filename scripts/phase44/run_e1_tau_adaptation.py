#!/usr/bin/env python3
"""E1 · hurt/OOD 切片上扫 τ：测「换数据集后重标定阈值」能否救回。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase32_common import eval_main_path, m2_head_ready, write_status  # noqa: E402
from _phase44_common import (  # noqa: E402
    OOD_PROXY_SLICES,
    TAU_SWEEP,
    TRANSFER_THR,
    write_phase44_result,
)
from boundary_budget import make_structure_budget_fn  # noqa: E402
from dataset_registry import load_slice  # noqa: E402
from phase23._phase23_common import CAP, MIN_N  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def best_tau_row(sweep: list[dict]) -> dict:
    return max(sweep, key=lambda r: (r["accuracy"], -(r.get("fallback_rate") or 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--slices", nargs="*", default=None, help="默认 OOD_PROXY_SLICES")
    args = ap.parse_args()

    if not m2_head_ready():
        raise SystemExit("缺少 M2 head；先跑 phase32/run_t0_ensure_m2_lite.py 或 A800 phase10")

    slice_ids = args.slices or OOD_PROXY_SLICES
    t0 = time.time()

    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(
        model, tokenizer, device, profile,
    )
    struct_only = make_structure_budget_fn(min_n=MIN_N, cap=CAP)

    by_slice = {}
    for sid in slice_ids:
        write_status({"phase": 44, "experiment": "e1", "current_slice": sid})
        spec, samples = load_slice(sid)
        if args.max_samples:
            samples = samples[: args.max_samples]
        main = eval_main_path(
            model, tokenizer, samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_only,
        )
        main_acc = main["accuracy"]

        sweep = []
        for thr in TAU_SWEEP:
            row = eval_confidence_fallback(
                head, model, tokenizer, samples, device=device, seed=args.seed,
                profile=profile, struct_floor=struct_floor, knn_floor=knn_floor,
                knn_thr=knn_thr, pfn=pfn, fallback_thr=thr,
            )
            sweep.append({
                "fallback_thr": thr,
                "accuracy": row["accuracy"],
                "delta_pp": round((row["accuracy"] - main_acc) * 100, 3),
                "fallback_rate": row.get("fallback_rate"),
            })

        frozen = next(s for s in sweep if abs(s["fallback_thr"] - TRANSFER_THR) < 1e-6)
        best = best_tau_row(sweep)
        by_slice[sid] = {
            "label": spec.get("label"),
            "category": spec.get("category"),
            "n_samples": len(samples),
            "main_acc": main_acc,
            "frozen_thr": frozen,
            "best_thr": best,
            "tau_gain_pp": round((best["accuracy"] - frozen["accuracy"]) * 100, 3),
            "sweep": sweep,
            "rescued": best["delta_pp"] > frozen["delta_pp"] + 0.5,
        }

    rescued = [sid for sid, r in by_slice.items() if r["rescued"]]
    payload = {
        "experiment_id": "e1_tau_adaptation",
        "title": "E1 · OOD 切片 τ 适配",
        "device": str(device),
        "seed": args.seed,
        "duration_sec": round(time.time() - t0, 2),
        "tau_sweep": TAU_SWEEP,
        "by_slice": by_slice,
        "rescued_slices": rescued,
        "insight": (
            "若 best_thr ≠ 0.48 且 tau_gain_pp>0，说明换数据集后应重标定阈值；"
            "若 best_thr≈0.99（永不回退）最优，说明该集应禁用通解回退。"
        ),
        "ok": True,
    }
    path = write_phase44_result("e1_tau_adaptation", payload)
    print(f"Wrote {path}; rescued {len(rescued)}/{len(slice_ids)}: {rescued}")


if __name__ == "__main__":
    main()
