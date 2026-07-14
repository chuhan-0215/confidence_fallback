"""Phase 45 · AlienBench_v1 陌生题库通解外推验证。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

PHASE45_OUT = ROOT / "results" / "phase45"
TRANSFER_THR = 0.48

ALIEN_SLICE_IDS = [
    "alien_full",
    "alien_star",
    "alien_ladder",
    "alien_hourglass",
    "alien_tree",
]


def write_phase45_result(eid: str, payload: dict) -> Path:
    from datetime import datetime, timezone

    PHASE45_OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {**payload, "finished_at": payload.get("finished_at") or ts}
    for name in (f"{eid}_latest.json", f"{eid}.json"):
        p = PHASE45_OUT / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return PHASE45_OUT / f"{eid}_latest.json"


def rollup_transfer_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    deltas = [r["delta_pp"] for r in rows if r.get("delta_pp") is not None]
    helps = sum(1 for d in deltas if d > 0.5)
    hurts = sum(1 for d in deltas if d < -0.5)
    return {
        "slice_count": len(rows),
        "transfer_helps_count": helps,
        "transfer_hurts_count": hurts,
        "transfer_neutral_count": len(rows) - helps - hurts,
        "mean_delta_pp": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "mean_main_acc": round(sum(r["main_acc"] for r in rows) / len(rows), 4),
        "mean_transfer_acc": round(
            sum(r["transfer_acc"] for r in rows if r.get("transfer_acc") is not None)
            / max(1, sum(1 for r in rows if r.get("transfer_acc") is not None)),
            4,
        ),
        "mean_fallback_rate": round(
            sum(r.get("fallback_rate") or 0 for r in rows) / len(rows), 4
        ),
    }


__all__ = [
    "ALIEN_SLICE_IDS",
    "PHASE45_OUT",
    "ROOT",
    "TRANSFER_THR",
    "rollup_transfer_rows",
    "write_phase45_result",
]
