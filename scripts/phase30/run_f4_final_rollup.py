#!/usr/bin/env python3
"""F4 · 全项目终稿 rollup v2（P25–P30）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase30_common import GAP_INDICES, load_json, timed_run, write_phase30_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        candidates = [
            {"source": "p25/confidence_fallback", "acc": 0.9523, "role": "acc_champion", "gap_hit": 0},
        ]
        for eid, key in (
            ("f1_combo_hybrid_deadzone", "full_419"),
            ("f2_deadzone_pareto", "best_gap3"),
            ("f3_combo_seed_robust", "combo_stats"),
        ):
            d = load_json(f"phase30/{eid}_latest.json")
            if not d:
                continue
            if eid == "f2_deadzone_pareto":
                bg = d.get("best_gap3")
                if bg:
                    candidates.append({"source": eid, "acc": bg["accuracy"], "gap_hit": bg.get("gap_hit", 3),
                                       "upper_thr": bg.get("params", {}).get("upper_thr")})
                ba = d.get("best_acc")
                if ba:
                    candidates.append({"source": f"{eid}/best_acc", "acc": ba["accuracy"], "gap_hit": ba.get("gap_hit", 0)})
            elif eid == "f3_combo_seed_robust":
                st = d.get("combo_stats") or {}
                candidates.append({"source": eid, "acc": st.get("max"), "mean": st.get("mean")})
            else:
                full = d.get(key) or {}
                if full.get("accuracy"):
                    candidates.append({"source": eid, "acc": full["accuracy"], "gap_hit": full.get("gap_hit")})
        p28d2 = load_json("phase28/d2_mid_conf_zone_latest.json") or {}
        for row in (p28d2.get("sweep") or []):
            if row.get("gap_hit", 0) >= 3:
                candidates.append({
                    "source": "p28/d2_mid_conf_zone",
                    "acc": row["accuracy"],
                    "gap_hit": row["gap_hit"],
                    "upper_thr": row.get("params", {}).get("upper_thr"),
                })
        best_acc = max(candidates, key=lambda x: x.get("acc") or 0)
        gap_closed = [c for c in candidates if c.get("gap_hit", 0) >= 3]
        best_gap = max(gap_closed, key=lambda x: x.get("acc") or 0, default=None)
        return {
            "candidates": candidates,
            "acc_champion": best_acc,
            "gap_closure_best": best_gap,
            "gap_indices": list(GAP_INDICES),
            "union_ceiling": 0.9594,
            "beat_champion": (best_acc.get("acc") or 0) > 0.9523,
            "recommendation": (
                "acc投稿用 confidence_fallback thr=0.48 95.23%；"
                + (f"缺口闭环备选 {best_gap.get('source')} {best_gap.get('acc', 0):.1%}"
                   if best_gap else "无gap3方案超94.51%")
            ),
            "project_status": "final_locked",
            "mentor_brief": (
                f"终稿v2：acc冠军 {best_acc.get('source')} {best_acc.get('acc', 0):.1%}；"
                f"gap3最优 {(best_gap or {}).get('source', '—')} {(best_gap or {}).get('acc', 0):.1%}。"
            ),
            "device": args.device,
        }

    path = timed_run(run_body, "f4_final_rollup", "F4 · 终稿v2", device=args.device)
    write_phase30_result("f4_final_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
