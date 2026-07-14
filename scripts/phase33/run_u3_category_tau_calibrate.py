#!/usr/bin/env python3
"""U3 · 按 category 标定 τ 上界（val 扫 grid，test 验证）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase33_common import (  # noqa: E402
    TAU_GRID,
    TRANSFER_THR,
    eval_main_path,
    m2_head_ready,
    split_val_test,
    unique_slice_ids,
    write_phase33_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit(f"缺少 M2 head: {ROOT / 'results/phase10/m2_enough_stop_head.pt'}")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, struct_floor, knn_floor, knn_thr, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    cat_val: dict[str, list] = defaultdict(list)
    cat_test: dict[str, list] = defaultdict(list)
    for sid in unique_slice_ids():
        meta, samples = load_slice(sid)
        val, test = split_val_test(samples, val_ratio=0.2, seed=43)
        cat = meta.get("category") or "unknown"
        cat_val[cat].extend(val)
        cat_test[cat].extend(test)

    cat_optimal: dict[str, dict] = {}
    for cat, val_samples in cat_val.items():
        best_t, best_acc = TRANSFER_THR, -1.0
        trials = []
        for thr in TAU_GRID:
            row = eval_confidence_fallback(
                head, model, tokenizer, val_samples, device=device, seed=args.seed, profile=profile,
                struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
                fallback_thr=thr,
            )
            trials.append({"threshold": thr, "val_acc": row["accuracy"]})
            if row["accuracy"] > best_acc:
                best_acc = row["accuracy"]
                best_t = thr
        cat_optimal[cat] = {"best_threshold": best_t, "val_acc": best_acc, "trials": trials}

    # global optimal on pooled val
    pooled_val = [s for samples in cat_val.values() for s in samples]
    global_best_t, global_best_acc = TRANSFER_THR, -1.0
    global_trials = []
    for thr in TAU_GRID:
        row = eval_confidence_fallback(
            head, model, tokenizer, pooled_val, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=thr,
        )
        global_trials.append({"threshold": thr, "val_acc": row["accuracy"]})
        if row["accuracy"] > global_best_acc:
            global_best_acc = row["accuracy"]
            global_best_t = thr

    test_results = {}
    for cat, test_samples in cat_test.items():
        if not test_samples:
            continue
        main_row = eval_main_path(
            model, tokenizer, test_samples, device=device, seed=args.seed,
            profile=profile, struct_floor=struct_floor,
        )
        frozen = eval_confidence_fallback(
            head, model, tokenizer, test_samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=TRANSFER_THR,
        )
        cat_t = cat_optimal[cat]["best_threshold"]
        adapted = eval_confidence_fallback(
            head, model, tokenizer, test_samples, device=device, seed=args.seed, profile=profile,
            struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
            fallback_thr=cat_t,
        )
        test_results[cat] = {
            "n_test": len(test_samples),
            "main_acc": main_row["accuracy"],
            "frozen_tau_048": frozen["accuracy"],
            "category_optimal_tau": cat_t,
            "category_optimal_acc": adapted["accuracy"],
            "delta_frozen_vs_main_pp": round((frozen["accuracy"] - main_row["accuracy"]) * 100, 2),
            "delta_adapted_vs_frozen_pp": round((adapted["accuracy"] - frozen["accuracy"]) * 100, 2),
        }

    pooled_test = [s for samples in cat_test.values() for s in samples]
    global_adapted = eval_confidence_fallback(
        head, model, tokenizer, pooled_test, device=device, seed=args.seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        fallback_thr=global_best_t,
    )
    frozen_all = eval_confidence_fallback(
        head, model, tokenizer, pooled_test, device=device, seed=args.seed, profile=profile,
        struct_floor=struct_floor, knn_floor=knn_floor, knn_thr=knn_thr, pfn=pfn,
        fallback_thr=TRANSFER_THR,
    )

    payload = {
        "experiment_id": "u3_category_tau_calibrate",
        "title": "U3 · 按 category 标定 τ",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "tau_grid": TAU_GRID,
        "frozen_tau": TRANSFER_THR,
        "category_optimal": cat_optimal,
        "global_optimal": {"best_threshold": global_best_t, "val_acc": global_best_acc, "trials": global_trials},
        "test_by_category": test_results,
        "pooled_test": {
            "n": len(pooled_test),
            "frozen_acc": frozen_all["accuracy"],
            "global_optimal_acc": global_adapted["accuracy"],
            "global_optimal_tau": global_best_t,
        },
    }
    write_phase33_result("u3_category_tau_calibrate", payload)
    print(json.dumps(payload["pooled_test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
