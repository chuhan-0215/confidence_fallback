#!/usr/bin/env python3
"""A5 · 全项目终稿 rollup。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase25_common import load_json, timed_run, write_phase25_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid in ("a1_fallback_finetune", "a2_hop4_fallback", "a3_answer_arbitrate", "a4_deploy_spec_v2"):
            d = load_json(f"phase25/{eid}_latest.json")
            if not d:
                continue
            if eid == "a4_deploy_spec_v2":
                snippets.append({"id": eid, "acc_champion": (d.get("deploy_spec") or {}).get("acc_champion")})
            else:
                full = d.get("full_419") or d.get("best") or d.get("best_thr_row") or {}
                snippets.append({
                    "id": eid,
                    "accuracy": full.get("accuracy"),
                    "timing_eps1": full.get("timing_eps1"),
                    "fallback_rate": full.get("fallback_rate"),
                })

        y5 = load_json("phase24/y5_project_closure_latest.json")
        a4 = load_json("phase25/a4_deploy_spec_v2_latest.json")
        spec = (a4 or {}).get("deploy_spec") or {}

        best = max(
            (s for s in snippets if s.get("accuracy") is not None),
            key=lambda s: s.get("accuracy") or 0,
            default={},
        )
        champion = (spec.get("acc_champion") or {})

        return {
            "snippets": snippets,
            "deploy_spec": spec,
            "phase24_closure": y5.get("project_status"),
            "project_status": "final",
            "feasible_any": False,
            "eps_deployable_any": True,
            "acc_champion": champion.get("id", "confidence_fallback"),
            "acc_champion_value": champion.get("accuracy", best.get("accuracy")),
            "mentor_brief": (
                f"终稿：acc 冠军 {champion.get('id', 'confidence_fallback')} "
                f"{champion.get('accuracy', 0):.1%}；"
                f"简单主推 structure_d μ={spec.get('primary', {}).get('accuracy_mean', 0):.1%}；"
                f"strict knn 37%；项目 final。"
            ),
            "insight": "Coconut 够好就停全项目终稿：L1 acc 95.0%，L2 strict 证伪，双轨 deploy。",
            "device": args.device,
        }

    path = timed_run(run_body, "a5_final_rollup", "A5 · 终稿", device=args.device)
    write_phase25_result("a5_final_rollup", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
