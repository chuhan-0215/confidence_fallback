"""Phase 17 · 修正 rollup bug + 最终定稿（CPU only，无需 GPU）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PHASE17_OUT = ROOT / "results" / "phase17"
FIXED_3_ACC = 0.863
TIMING_FLOOR = 0.5


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def load_json(*rels: str) -> dict:
    for rel in rels:
        for base in (ROOT / "outbox/results/from_a800", ROOT / "results"):
            p = base / rel
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
    return {}


def m3_necessity_pass(m3: dict) -> bool:
    """pct_improved=0 是合法值，不能用 `or 1`。"""
    ab = m3.get("ablation") or {}
    improved = ab.get("pct_improved")
    degraded = ab.get("pct_degraded")
    if improved is None:
        improved = 1.0
    if degraded is None:
        degraded = 0.0
    return improved < 0.05 and degraded > 0.05


def is_deployable_row(row: dict) -> bool:
    if not row:
        return False
    return (
        row.get("accuracy", 0) >= FIXED_3_ACC
        and (row.get("mean_stop_n") or 99) <= 4.5
        and row.get("params", {}).get("uses_oracle", False) is not True
    )


def write_result(eid: str, payload: dict) -> Path:
    PHASE17_OUT.mkdir(parents=True, exist_ok=True)
    path = PHASE17_OUT / f"{eid}_latest.json"
    payload.setdefault("ok", True)
    payload.setdefault("finished_at", utc_now())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
