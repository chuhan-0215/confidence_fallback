"""Unified phase result JSON writer."""
from __future__ import annotations

import json
from pathlib import Path

from phase4._phase4_common import utc_now


def write_phase_result(phase: int, experiment_id: str, payload: dict) -> Path:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / f"phase{phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{experiment_id}_latest.json"
    payload.setdefault("ok", True)
    payload.setdefault("finished_at", utc_now())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
