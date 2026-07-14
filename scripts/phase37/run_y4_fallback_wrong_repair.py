#!/usr/bin/env python3
"""Y4 · fallback_wrong 7 题修复探测。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase25"))
sys.path.insert(0, str(ROOT / "scripts" / "phase34"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fallback_eval import eval_confidence_fallback, setup_fallback_stack  # noqa: E402
from _phase34_common import TRANSFER_THR, eval_agreement_lock, eval_tri_zone  # noqa: E402
from _phase37_common import FALLBACK_WRONG_IDX, LOCKED_TRI_ZONE, m2_head_ready, write_phase37_result  # noqa: E402
from phase23._phase23_common import load_full_dataset  # noqa: E402
from phase4._phase4_common import load_model_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    if not m2_head_ready():
        raise SystemExit("缺少 M2 head")

    t0 = time.time()
    model, tokenizer, device, profile = load_model_bundle(args.device)
    head, sf, kf, kt, pfn = setup_fallback_stack(model, tokenizer, device, profile)
    samples = load_full_dataset()
    sub = [samples[i] for i in FALLBACK_WRONG_IDX if i < len(samples)]
    kw = dict(head=head, model=model, tokenizer=tokenizer, device=device, seed=args.seed, profile=profile,
              struct_floor=sf, knn_floor=kf, knn_thr=kt, pfn=pfn)

    policies = {
        "baseline": eval_confidence_fallback(sub, **kw, fallback_thr=TRANSFER_THR),
        "agreement_lock": eval_agreement_lock(sub, **kw, fallback_thr=TRANSFER_THR, hop4_only=False),
        "tri_zone": eval_tri_zone(sub, **kw, t_low=LOCKED_TRI_ZONE[0], t_mid=LOCKED_TRI_ZONE[1]),
    }
    full_baseline = eval_confidence_fallback(samples, **kw, fallback_thr=TRANSFER_THR)

    payload = {
        "experiment_id": "y4_fallback_wrong_repair",
        "title": "Y4 · fallback_wrong 7 题修复",
        "device": str(device),
        "duration_sec": round(time.time() - t0, 2),
        "indices": list(FALLBACK_WRONG_IDX),
        "policies_on_subset": policies,
        "full419_baseline_acc": full_baseline["accuracy"],
        "p25_target": 0.9523,
        "gap_pp": round((0.9523 - full_baseline["accuracy"]) * 100, 2),
        "repairable_by_tri_zone": policies["tri_zone"]["accuracy"] > policies["baseline"]["accuracy"],
    }
    write_phase37_result("y4_fallback_wrong_repair", payload)
    print(json.dumps(policies, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
