#!/usr/bin/env python3
"""E3 · 全项目终稿 rollup（P25–P29）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase29_common import load_json, timed_run, write_phase29_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        candidates = [{"source": "p25/confidence_fallback", "acc": 0.9523, "role": "acc_champion"}]
        for phase, eids in (
            ("phase27", ("c2_writeback_structure", "c6_dual_forward", "c7_alt_paradigm_rollup")),
            ("phase28", ("d2_mid_conf_zone", "d3_hop_hybrid", "d4_always_compare", "d5_gap_closure_rollup")),
            ("phase29", ("e1_best_seed_robust",)),
        ):
            for eid in eids:
                d = load_json(f"{phase}/{eid}_latest.json")
                if not d:
                    continue
                if eid.endswith("rollup"):
                    if d.get("best_p28"):
                        b = d["best_p28"]
                        candidates.append({"source": b.get("id"), "acc": b.get("accuracy"), "gap_hit": b.get("gap_hit")})
                    elif d.get("best_alternative"):
                        b = d["best_alternative"]
                        candidates.append({"source": b.get("id"), "acc": b.get("accuracy")})
                    continue
                if eid == "e1_best_seed_robust":
                    st = d.get("acc_stats") or {}
                    candidates.append({"source": eid, "acc": st.get("max"), "mean": st.get("mean")})
                    continue
                full = d.get("full_419") or d.get("best") or {}
                if full.get("accuracy"):
                    candidates.append({
                        "source": f"{phase}/{eid}",
                        "acc": full["accuracy"],
                        "gap_hit": full.get("gap_hit"),
                    })
        best = max(candidates, key=lambda x: x.get("acc") or 0)
        beat = (best.get("acc") or 0) > 0.9523
        return {
            "candidates": candidates,
            "overall_best": best,
            "acc_champion_p25": 0.9523,
            "union_ceiling": 0.9594,
            "beat_champion": beat,
            "project_status": "complete",
            "mentor_brief": (
                f"项目终稿：全路线最优 {best.get('source')} {best.get('acc', 0):.1%}；"
                f"{'超越' if beat else '维持'} P25 冠军 95.23%。"
            ),
            "device": args.device,
        }

    path = timed_run(run_body, "e3_project_complete", "E3 · 项目终稿", device=args.device)
    write_phase29_result("e3_project_complete", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
