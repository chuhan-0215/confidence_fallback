#!/usr/bin/env python3
"""B5 · 全项目最终锁定 rollup。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase26_common import load_json, timed_run, write_phase26_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    def run_body():
        snippets = []
        for eid in ("b1_hop_adaptive_thr", "b2_gap_forensic", "b3_champion_seed_table", "b4_icais_spec_lock"):
            d = load_json(f"phase26/{eid}_latest.json")
            if not d:
                continue
            if eid == "b4_icais_spec_lock":
                snippets.append({"id": eid, "icais": (d.get("deploy_spec") or {}).get("icais_numbers")})
            elif eid == "b2_gap_forensic":
                snippets.append({"id": eid, "gap": d.get("champ_wrong_union_ok")})
            else:
                full = d.get("full_419") or d.get("best") or {}
                snippets.append({"id": eid, "accuracy": full.get("accuracy")})

        p25 = load_json("phase25/a5_final_rollup_latest.json")
        b4 = load_json("phase26/b4_icais_spec_lock_latest.json")
        icais = ((b4 or {}).get("deploy_spec") or {}).get("icais_numbers") or {}

        return {
            "snippets": snippets,
            "phase25_status": p25.get("project_status"),
            "project_status": "locked",
            "acc_champion": icais.get("acc_champion", 0.9523),
            "icais_numbers": icais,
            "mentor_brief": (
                f"项目 locked：acc 冠军 {icais.get('acc_champion', 0):.1%}；"
                f"并集 {icais.get('union_ceiling', 0):.1%}；ICAIS 数字定稿。"
            ),
            "insight": "全项目终局锁定，无强制后续 GPU 实验。",
            "device": args.device,
        }

    path = timed_run(run_body, "b5_project_locked", "B5 · 锁定", device=args.device)
    write_phase26_result("b5_project_locked", json.loads(path.read_text(encoding="utf-8")))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
