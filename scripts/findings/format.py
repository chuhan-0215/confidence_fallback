"""HTML/number formatting helpers."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTBOX = ROOT / "outbox" / "results" / "from_a800"


def _candidate_paths(name: str) -> list[Path]:
    paths: list[Path] = []
    if "/" in name:
        phase_dir, fname = name.split("/", 1)
        for base in (RESULTS, OUTBOX):
            paths.append(base / phase_dir / fname)
            paths.append(base / "from_a800" / phase_dir / fname)
    paths.append(RESULTS / name)
    paths.append(RESULTS / "from_a800" / name)
    if "/" in name:
        paths.append(OUTBOX / name.split("/", 1)[0] / name.split("/", 1)[1])
    return paths


def load_json(name: str) -> dict | None:
    for path in _candidate_paths(name):
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return None


def fmt_pct(x) -> str:
    if x is None:
        return "—"
    return f"{float(x) * 100:.1f}%"


def fmt_step(x) -> str:
    if x is None:
        return "—"
    v = float(x)
    return str(int(v)) if v == int(v) else str(v)


def esc(text) -> str:
    return html.escape(str(text or ""), quote=True)


