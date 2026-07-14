#!/usr/bin/env python3
"""Y5 · 全项目收官 rollup。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase24_common import load_json, timed_run, write_phase24_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid in ("y1_hop4_budget", "y2_confidence_fallback", "y3_error_taxonomy", "y4_deploy_spec"):
            d = load_json(f"phase24/{eid}_latest.json")
            if not d:
                continue
            if eid == "y4_deploy_spec":
                snippets.append({"id": eid, "deploy_spec": d.get("deploy_spec")})
            elif eid == "y3_error_taxonomy":
                snippets.append({
                    "id": eid, "struct_wrong": d.get("struct_wrong"),
                    "both_wrong": d.get("both_wrong"), "union_miss": d.get("union_miss"),
                })
            else:
                full = d.get("full_419") or d.get("best") or {}
                snippets.append({"id": eid, "accuracy": full.get("accuracy"), "timing_eps1": full.get("timing_eps1")})

        p23 = load_json("phase23/z6_final_manifest_latest.json")
        y4 = load_json("phase24/y4_deploy_spec_latest.json")
        spec = (y4 or {}).get("deploy_spec") or {}

        best_y = max(
            (s for s in snippets if s.get("accuracy") is not None),
            key=lambda s: s.get("accuracy") or 0,
            default={},
        )

        brief = (
            f"全项目收官：主推 structure_d μ={spec.get('primary', {}).get('accuracy_mean', 0):.1%}；"
            f"P24 最优 {best_y.get('id', '—')} {best_y.get('accuracy', 0):.1%}；"
            f"strict timing knn 37%；ε-deploy 50.7%。"
        )

        return {
            "snippets": snippets,
            "deploy_spec": spec,
            "phase23_manifest": p23.get("deploy_manifest"),
            "project_status": "closure",
            "feasible_any": False,
            "eps_deployable_any": True,
            "beat_p23_best": (best_y.get("accuracy") or 0) > 0.9475,
            "mentor_brief": brief,
            "insight": "Coconut 够好就停全项目收官：L1 达成，L2 strict 证伪，ε-deploy 作 stretch。",
            "device": args.device,
        }

    path = timed_run(run_body, "y5_project_closure", "Y5 · 收官", device=args.device)
    write_phase24_result("y5_project_closure", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
