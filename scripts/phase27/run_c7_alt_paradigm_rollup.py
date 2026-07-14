#!/usr/bin/env python3
"""C7 · 替代范式汇总 + 与冠军对照。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase27_common import load_json, timed_run, write_phase27_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid in ("c1_depth_vote", "c2_writeback_structure", "c3_stability_gate",
                    "c4_convergence_stop", "c5_multi_seed_vote", "c6_dual_forward"):
            d = load_json(f"phase27/{eid}_latest.json")
            if not d:
                continue
            full = d.get("full_419") or d.get("best") or {}
            snippets.append({
                "id": eid,
                "accuracy": full.get("accuracy"),
                "mean_stop_n": full.get("mean_stop_n"),
                "mode": (full.get("params") or {}).get("mode"),
            })
        p25 = load_json("phase25/a1_fallback_finetune_latest.json")
        champ = (p25.get("best_thr_row") or {}).get("accuracy", 0.9523)
        best_alt = max((s for s in snippets if s.get("accuracy")), key=lambda s: s["accuracy"], default={})
        beat = (best_alt.get("accuracy") or 0) > champ
        return {
            "snippets": snippets,
            "champion_acc": champ,
            "best_alternative": best_alt,
            "beat_champion": beat,
            "mentor_brief": (
                f"C7 替代范式：最优 {best_alt.get('id', '—')} {best_alt.get('accuracy', 0):.1%}；"
                f"冠军 {champ:.1%}；{'超越' if beat else '未超越'}。"
            ),
            "insight": "非 Stop Head 路线全景对照。",
            "device": args.device,
        }

    path = timed_run(run_body, "c7_alt_paradigm_rollup", "C7 · 替代汇总", device=args.device)
    write_phase27_result("c7_alt_paradigm_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
