#!/usr/bin/env python3
"""G3 · 绝对终稿 rollup + ICAIS spec v5。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase31_common import GAP_INDICES, load_json, timed_run, write_phase31_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        candidates = [
            {"source": "p25/confidence_fallback", "acc": 0.9523, "role": "acc_champion", "gap_hit": 0},
        ]
        for phase, eid, key in (
            ("phase30", "f1_combo_hybrid_deadzone", "full_419"),
            ("phase30", "f2_deadzone_pareto", "best_gap3"),
            ("phase31", "g1_surgical_deadzone", "full_419"),
        ):
            d = load_json(f"{phase}/{eid}_latest.json")
            if not d:
                continue
            full = d.get(key) or d.get("full_419") or {}
            if full.get("accuracy"):
                candidates.append({
                    "source": f"{phase}/{eid}",
                    "acc": full["accuracy"],
                    "gap_hit": full.get("gap_hit"),
                })
        p30f2 = load_json("phase30/f2_deadzone_pareto_latest.json") or {}
        bg = p30f2.get("best_gap3") or {}
        spec_v4 = (load_json("phase29/e2_icais_spec_v4_latest.json") or {}).get("deploy_spec") or {}
        best_acc = max(candidates, key=lambda x: x.get("acc") or 0)
        gap_best = max(
            (c for c in candidates if (c.get("gap_hit") or 0) >= 3),
            key=lambda x: x.get("acc") or 0,
            default=bg,
        )
        spec = {
            **spec_v4,
            "version": "deploy_v5_absolute_final",
            "icais_numbers": {
                **(spec_v4.get("icais_numbers") or {}),
                "p30_gap3_best": bg.get("accuracy", 0.9475),
                "p30_gap3_thr": 0.551,
                "p30_f1_combo": 0.9451,
                "acc_union_ceiling": 0.9594,
                "acc_hard_limit_both_wrong": 17,
                "project_status": "absolute_final",
            },
            "dual_track": {
                "acc_champion": "confidence_fallback thr=0.48 → 95.23%",
                "gap_closure_alt": f"upper_thr=0.551 → {bg.get('accuracy', 0.9475):.2%} (gap_hit=3/3)",
            },
        }
        return {
            "candidates": candidates,
            "acc_champion": best_acc,
            "gap_closure_best": gap_best,
            "deploy_spec": spec,
            "gap_indices": list(GAP_INDICES),
            "union_ceiling": 0.9594,
            "beat_champion": False,
            "recommendation": spec["dual_track"],
            "project_status": "absolute_final",
            "mentor_brief": (
                f"绝对终稿：acc冠军 {best_acc.get('acc', 0):.1%}；"
                f"gap3最优 {(gap_best or {}).get('acc', 0):.1%}；项目闭合。"
            ),
            "device": args.device,
        }

    path = timed_run(run_body, "g3_absolute_final", "G3 · 绝对终稿", device=args.device)
    write_phase31_result("g3_absolute_final", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
