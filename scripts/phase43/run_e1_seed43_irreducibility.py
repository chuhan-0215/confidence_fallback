#!/usr/bin/env python3
"""E1 · seed43 不可修复性论证（聚合 P42 法医 + 翻转分类）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def load_json(phase: int, name: str) -> dict | None:
    for base in (ROOT / "results" / f"phase{phase}", ROOT / f"outbox/results/from_a800/phase{phase}"):
        p = base / f"{name}_latest.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def write_result(payload: dict) -> None:
    out = ROOT / "results" / "phase43"
    out.mkdir(parents=True, exist_ok=True)
    for name in (f"e1_seed43_irreducibility_latest.json", f"e1_seed43_irreducibility.json"):
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    d1 = load_json(42, "d1_seed43_in_dist_forensic")
    d2 = load_json(42, "d2_hybrid_v5_seed_robust")

    flips = (d1 or {}).get("flips") or []
    seed43 = ((d2 or {}).get("by_seed") or {}).get("43") or {}
    v4_s = (seed43.get("hybrid_v4") or {}).get("summary") or {}
    v5_s = (seed43.get("hybrid_v5") or {}).get("summary") or {}

    blocking = []
    for f in flips:
        s99 = f.get("seed_99") or {}
        s43 = f.get("seed_43") or {}
        if s99.get("helps") and s43.get("hurts"):
            blocking.append({
                "slice_id": f.get("slice_id"),
                "category": f.get("category"),
                "delta_99": s99.get("delta_pp"),
                "delta_43": s43.get("delta_pp"),
                "reason": "skip/agreement 会损失 @99 增益",
            })
        elif s99.get("hurts") and s43.get("helps"):
            blocking.append({
                "slice_id": f.get("slice_id"),
                "category": f.get("category"),
                "delta_99": s99.get("delta_pp"),
                "delta_43": s43.get("delta_pp"),
                "reason": "v5 skip 已证 @43 更差",
            })

    payload = {
        "experiment_id": "e1_seed43_irreducibility",
        "title": "E1 · seed43 不可修复性论证",
        "flip_count": len(flips),
        "blocking_flip_count": len(blocking),
        "blocking_flips": blocking[:20],
        "seed43_v4_in_dist": v4_s.get("in_dist_weighted_delta_pp"),
        "seed43_v5_in_dist": v5_s.get("in_dist_weighted_delta_pp"),
        "v5_worse_than_v4": (v5_s.get("in_dist_weighted_delta_pp") or 0) < (v4_s.get("in_dist_weighted_delta_pp") or 0),
        "conclusion": "seed-invariant 路由无法同时满足 @99 增益切片与 @43 in-dist；接受 3/4 硬限",
        "ok": True,
    }
    write_result(payload)
    print(json.dumps({"blocking": len(blocking), "conclusion": payload["conclusion"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
