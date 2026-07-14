"""Shared helpers for Phase 4 (non-track) experiments."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "model"))

from evaluate_coconut import load_coconut_model, load_dataset, resolve_device  # noqa: E402
from eval_profile import parse_eval_profile  # noqa: E402
from run_experiment import ensure_checkpoint  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model_bundle(device_name: str = "cuda"):
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ensure_checkpoint(ROOT / "checkpoints" / "checkpoint_300")
    device = resolve_device(device_name)
    model, tokenizer = load_coconut_model(checkpoint, config_path, device)
    profile = parse_eval_profile(None)
    return model, tokenizer, device, profile


def write_result(experiment_id: str, payload: dict, status_file: Optional[Path] = None) -> Path:
    out_dir = ROOT / "results" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{experiment_id}_latest.json"
    payload.setdefault("ok", True)
    payload.setdefault("finished_at", utc_now())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if status_file:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(json.dumps({"running": False, "updated_at": utc_now()}, indent=2))
    return path


def timed_run(fn: Callable[[], dict], experiment_id: str, title: str, **meta) -> Path:
    t0 = time.time()
    body = fn()
    body.update(
        {
            "experiment_id": experiment_id,
            "title": title,
            "duration_sec": round(time.time() - t0, 2),
            **meta,
        }
    )
    return write_result(experiment_id, body)
