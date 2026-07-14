#!/usr/bin/env python3
"""E3 · 八 seed 面板 rollup（v4）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import setup_fallback_stack  # noqa: E402
from _phase43_common import (  # noqa: E402
    PANEL_SEEDS,
    dual_ok,
    eval_hybrid_v4_router,
    m2_head_ready,
    unique_slice_ids,
    write_phase43_result,
)
from dataset_registry import load_slice  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402
from shared.eval_paths import eval_main_path, rollup_slice_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)

    by_seed = {}
    for seed in PANEL_SEEDS:
        rows = []
        for sid in unique_slice_ids():
            meta, samples = load_slice(sid)
            main_row = eval_main_path(model, tokenizer, samples, device=device, seed=seed, profile=profile, struct_floor=sf)
            v4 = eval_hybrid_v4_router(head, model, tokenizer, samples, device=device, seed=seed, profile=profile,
                                       struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn, meta=meta)
            rows.append({
                "slice_id": sid, "category": meta.get("category"), "n_samples": len(samples),
                "main_acc": main_row["accuracy"], "policy_acc": v4["accuracy"],
                "delta_pp": round((v4["accuracy"] - main_row["accuracy"]) * 100, 2),
            })
        s = rollup_slice_rows(rows)
        by_seed[str(seed)] = {"summary": s, "dual_ok": dual_ok(s)}
        print(f"seed={seed} dual_ok={dual_ok(s)} in={s.get('in_dist_weighted_delta_pp')} ood={s.get('ood_weighted_delta_pp')}", flush=True)

    dual_ok_seeds = [int(k) for k, v in by_seed.items() if v["dual_ok"]]
    payload = {
        "experiment_id": "e3_seed_panel_rollup",
        "title": "E3 · 八 seed 面板",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "router": "hybrid_v4",
        "seeds": list(PANEL_SEEDS),
        "by_seed": by_seed,
        "dual_ok_count": len(dual_ok_seeds),
        "dual_ok_seeds": sorted(dual_ok_seeds),
        "ok": True,
    }
    write_phase43_result("e3_seed_panel_rollup", payload)
    print(json.dumps({"dual_ok": f"{len(dual_ok_seeds)}/{len(PANEL_SEEDS)}", "seeds": dual_ok_seeds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
