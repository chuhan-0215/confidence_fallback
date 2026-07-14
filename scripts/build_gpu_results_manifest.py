#!/usr/bin/env python3
"""Scan results/from_a800 (+ phase4) and build gpu_results_manifest.json for the lab UI."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
FROM_A800 = ROOT / "results" / "from_a800"
PHASE4_CPU = ROOT / "results" / "phase4"
PHASE5_CPU = ROOT / "results" / "phase5"
OUTBOX_RESULTS = ROOT / "outbox" / "results" / "from_a800"

OUT_PATH = ROOT / "results" / "gpu_results_manifest.json"

TRACK_RE = re.compile(r"^adaptive_stop_v(\d+)_latest\.json$")
PHASE4_RE = re.compile(r"^(x\d+_[a-z0-9_]+)_latest\.json$")
PHASE5_RE = re.compile(r"^(y\d+_[a-z0-9_]+)_latest\.json$")
PHASE6_RE = re.compile(r"^((?:y\d+|z\d+)_[a-z0-9_]+)_latest\.json$")
PHASE7_RE = PHASE6_RE
PHASE8_RE = PHASE6_RE
PHASE9_RE = PHASE6_RE
PHASE10_RE = re.compile(r"^(m[123]_[a-z0-9_]+)_latest\.json$")
PHASE11_RE = re.compile(r"^(j[1-5]_[a-z0-9_]+)_latest\.json$")
PHASE12_RE = re.compile(r"^(k[a-z0-9_]+)_latest\.json$")
PHASE13_RE = re.compile(r"^(t[a-z0-9_]+)_latest\.json$")
PHASE14_RE = re.compile(r"^(p[a-z0-9_]+)_latest\.json$")
PHASE15_RE = re.compile(r"^(q[a-z0-9_]+)_latest\.json$")
PHASE16_RE = re.compile(r"^(r[a-z0-9_]+)_latest\.json$")
PHASE17_RE = re.compile(r"^(s\d+_[a-z0-9_]+)_latest\.json$")
PHASE18_RE = re.compile(r"^(w\d+_[a-z0-9_]+)_latest\.json$")
PHASE19_RE = re.compile(r"^(u\d+_[a-z0-9_]+)_latest\.json$")
PHASE20_RE = re.compile(r"^(v\d+_[a-z0-9_]+)_latest\.json$")
PHASE21_RE = re.compile(r"^(w\d+_[a-z0-9_]+)_latest\.json$")
PHASE22_RE = re.compile(r"^(x\d+_[a-z0-9_]+)_latest\.json$")
PHASE23_RE = re.compile(r"^(z\d+_[a-z0-9_]+)_latest\.json$")
PHASE24_RE = re.compile(r"^(y\d+_[a-z0-9_]+)_latest\.json$")
PHASE25_RE = re.compile(r"^(a\d+_[a-z0-9_]+)_latest\.json$")
PHASE26_RE = re.compile(r"^(b\d+_[a-z0-9_]+)_latest\.json$")
PHASE27_RE = re.compile(r"^(c\d+_[a-z0-9_]+)_latest\.json$")
PHASE28_RE = re.compile(r"^(d\d+_[a-z0-9_]+)_latest\.json$")
PHASE29_RE = re.compile(r"^(e\d+_[a-z0-9_]+)_latest\.json$")
PHASE30_RE = re.compile(r"^(f\d+_[a-z0-9_]+)_latest\.json$")
PHASE31_RE = re.compile(r"^(g\d+_[a-z0-9_]+)_latest\.json$")
PHASE32_RE = re.compile(r"^(t\d+_[a-z0-9_]+)_latest\.json$")
PHASE33_RE = re.compile(r"^(u\d+_[a-z0-9_]+)_latest\.json$")
PHASE34_RE = re.compile(r"^(v\d+_[a-z0-9_]+)_latest\.json$")
PHASE35_RE = re.compile(r"^(w\d+_[a-z0-9_]+)_latest\.json$")
PHASE36_RE = re.compile(r"^(x\d+_[a-z0-9_]+)_latest\.json$")
PHASE37_RE = re.compile(r"^(y\d+_[a-z0-9_]+)_latest\.json$")
PHASE38_RE = re.compile(r"^(z\d+_[a-z0-9_]+)_latest\.json$")
PHASE39_RE = re.compile(r"^(a\d+_[a-z0-9_]+)_latest\.json$")
PHASE40_RE = re.compile(r"^(b\d+_[a-z0-9_]+)_latest\.json$")
PHASE41_RE = re.compile(r"^(c\d+_[a-z0-9_]+)_latest\.json$")
PHASE42_RE = re.compile(r"^(d\d+_[a-z0-9_]+)_latest\.json$")
PHASE43_RE = re.compile(r"^(e\d+_[a-z0-9_]+)_latest\.json$")
PHASE44_RE = re.compile(r"^(e\d+_[a-z0-9_]+)_latest\.json$")

GPU_PHASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("phase44", PHASE44_RE),
    ("phase43", PHASE43_RE),
    ("phase42", PHASE42_RE),
    ("phase41", PHASE41_RE),
    ("phase40", PHASE40_RE),
    ("phase39", PHASE39_RE),
    ("phase38", PHASE38_RE),
    ("phase37", PHASE37_RE),
    ("phase36", PHASE36_RE),
    ("phase35", PHASE35_RE),
    ("phase34", PHASE34_RE),
    ("phase33", PHASE33_RE),
    ("phase32", PHASE32_RE),
    ("phase31", PHASE31_RE),
    ("phase30", PHASE30_RE),
    ("phase29", PHASE29_RE),
    ("phase28", PHASE28_RE),
    ("phase27", PHASE27_RE),
    ("phase26", PHASE26_RE),
    ("phase25", PHASE25_RE),
    ("phase24", PHASE24_RE),
    ("phase23", PHASE23_RE),
    ("phase22", PHASE22_RE),
    ("phase21", PHASE21_RE),
    ("phase20", PHASE20_RE),
    ("phase19", PHASE19_RE),
    ("phase18", PHASE18_RE),
    ("phase17", PHASE17_RE),
    ("phase16", PHASE16_RE),
    ("phase15", PHASE15_RE),
    ("phase14", PHASE14_RE),
    ("phase13", PHASE13_RE),
    ("phase12", PHASE12_RE),
    ("phase11", PHASE11_RE),
    ("phase10", PHASE10_RE),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_metrics(data: dict) -> dict[str, Any]:
    s = data.get("summary") or {}
    acc = (
        s.get("main_strategy_accuracy")
        or s.get("joint_correctness_stop_accuracy")
        or s.get("correctness_stop_accuracy")
        or s.get("rich_stop_accuracy")
        or s.get("trained_stop_accuracy")
    )
    timing = (
        s.get("main_strategy_timing_acc")
        or s.get("joint_correctness_stop_timing_acc")
        or s.get("correctness_stop_timing_acc")
        or s.get("rich_stop_timing_acc")
        or s.get("trained_stop_timing_acc")
    )
    return {
        "accuracy": acc,
        "timing_acc": timing,
        "feasible": s.get("trainable_stop_feasible"),
        "strategy": s.get("best_learned_strategy"),
        "auto_route": s.get("auto_route_accuracy"),
        "fixed_3": s.get("fixed_3_accuracy"),
        "single_forward": s.get("single_forward"),
        "inference_probes": s.get("inference_probes"),
    }


def _gpu_phase_metrics(data: dict) -> dict[str, Any]:
    if data.get("primary_recommend"):
        rec = data["primary_recommend"]
        return {
            "accuracy": rec.get("accuracy"),
            "timing_acc": rec.get("timing"),
            "feasible": data.get("deployable_mvp") or rec.get("deployable_mvp"),
            "strategy": rec.get("name"),
        }
    if data.get("strategies") and not data.get("full_419"):
        best = max(
            data["strategies"],
            key=lambda s: (
                bool(s.get("deployable_mvp")),
                s.get("timing") or 0,
                s.get("accuracy") or 0,
            ),
            default=None,
        )
        if best:
            return {
                "accuracy": best.get("accuracy"),
                "timing_acc": best.get("timing"),
                "feasible": best.get("deployable_mvp") or data.get("deployable_mvp"),
                "strategy": best.get("name"),
            }
    full = data.get("full_419") or data.get("full_419_m2")
    test = data.get("test") or data.get("test_40pct")
    row = full if isinstance(full, dict) else None
    if not row and isinstance(test, dict):
        row = test
    if not row and data.get("variants"):
        variants = data["variants"]
        if isinstance(variants, dict):
            for key in ("best_deployable_mvp", "best_timing", "best_acc", "best"):
                v = variants.get(key) or {}
                row = v.get("full_419") or v.get("test_40pct")
                if row:
                    break
    if not row and data.get("best") and isinstance(data["best"], dict):
        if data["best"].get("accuracy") is not None:
            return {
                "accuracy": data["best"].get("accuracy"),
                "timing_acc": data["best"].get("timing") or data["best"].get("stop_timing_acc"),
                "feasible": data.get("feasible") or data.get("deployable_mvp"),
                "strategy": data["best"].get("id") or data["best"].get("strategy"),
            }
    if not row and data.get("candidates"):
        best = max(
            data["candidates"],
            key=lambda c: ((c.get("timing") or c.get("stop_timing_acc") or 0), c.get("accuracy") or 0),
            default=None,
        )
        if best:
            return {
                "accuracy": best.get("accuracy"),
                "timing_acc": best.get("timing") or best.get("stop_timing_acc"),
                "feasible": best.get("feasible") or data.get("feasible_any"),
                "strategy": best.get("id") or best.get("strategy"),
            }
    acc = (row or {}).get("accuracy")
    timing = (row or {}).get("stop_timing_acc")
    feasible = data.get("feasible")
    if feasible is None:
        feasible = data.get("deployable_mvp") or data.get("strict_feasible") or data.get("fully_proven")
    return {
        "accuracy": acc,
        "timing_acc": timing,
        "feasible": feasible,
        "strategy": (row or {}).get("strategy") or (data.get("best") or {}).get("id"),
    }


def _build_gpu_phase_entry(path: Path, data: dict, kind: str, exp_id: str) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    metrics = _gpu_phase_metrics(data)
    phase_num = kind.replace("phase", "")
    return {
        "key": exp_id,
        "kind": kind,
        "phase": int(phase_num) if phase_num.isdigit() else None,
        "track": None,
        "experiment_id": data.get("experiment_id") or exp_id,
        "title": data.get("title") or exp_id,
        "device": data.get("device") or "cuda",
        "sample_count": data.get("sample_count"),
        "duration_sec": data.get("duration_sec"),
        "finished_at": data.get("finished_at"),
        "source_file": rel,
        "source_mtime": mtime,
        "insight": data.get("insight") or data.get("mentor_brief"),
        "payload_keys": list(data.keys())[:12],
        **metrics,
    }


def _match_gpu_phase(path: Path, name: str) -> Optional[tuple[str, re.Match[str]]]:
    posix = path.as_posix()
    for kind, pattern in GPU_PHASE_PATTERNS:
        if f"/{kind}/" not in posix:
            continue
        m = pattern.match(name)
        if m:
            return kind, m
    return None


def _entry_from_json(path: Path) -> Optional[dict[str, Any]]:
    name = path.name
    gpu_phase = _match_gpu_phase(path, name)
    if gpu_phase:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        kind, m = gpu_phase
        return _build_gpu_phase_entry(path, data, kind, m.group(1))

    track_match = TRACK_RE.match(name)
    phase4_match = PHASE4_RE.match(name)
    phase5_match = PHASE5_RE.match(name)
    phase6_match = PHASE6_RE.match(name)
    phase7_match = PHASE7_RE.match(name)
    phase8_match = PHASE8_RE.match(name)
    phase9_match = PHASE9_RE.match(name)
    phase10_match = PHASE10_RE.match(name)
    phase11_match = PHASE11_RE.match(name)
    phase12_match = PHASE12_RE.match(name)
    phase13_match = PHASE13_RE.match(name)
    phase14_match = PHASE14_RE.match(name)
    phase15_match = PHASE15_RE.match(name)
    phase16_match = PHASE16_RE.match(name)
    in_phase6 = "phase6" in path.as_posix()
    in_phase7 = "phase7" in path.as_posix()
    in_phase8 = "phase8" in path.as_posix()
    in_phase9 = "phase9" in path.as_posix()
    in_phase10 = "phase10" in path.as_posix()
    in_phase11 = "phase11" in path.as_posix()
    in_phase12 = "phase12" in path.as_posix()
    in_phase13 = "phase13" in path.as_posix()
    in_phase14 = "phase14" in path.as_posix()
    in_phase15 = "phase15" in path.as_posix()
    in_phase16 = "phase16" in path.as_posix()
    if phase6_match and in_phase6:
        phase5_match = None
    if phase7_match and in_phase7:
        phase5_match = None
        phase6_match = None
    if phase8_match and in_phase8:
        phase5_match = None
        phase6_match = None
        phase7_match = None
    if phase9_match and in_phase9:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
    if phase10_match and in_phase10:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
    if phase11_match and in_phase11:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
        phase10_match = None
    if phase12_match and in_phase12:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
        phase10_match = None
        phase11_match = None
    if phase13_match and in_phase13:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
        phase10_match = None
        phase11_match = None
        phase12_match = None
    if phase14_match and in_phase14:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
        phase10_match = None
        phase11_match = None
        phase12_match = None
        phase13_match = None
    if phase15_match and in_phase15:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
        phase10_match = None
        phase11_match = None
        phase12_match = None
        phase13_match = None
        phase14_match = None
    if phase16_match and in_phase16:
        phase5_match = None
        phase6_match = None
        phase7_match = None
        phase8_match = None
        phase9_match = None
        phase10_match = None
        phase11_match = None
        phase12_match = None
        phase13_match = None
        phase14_match = None
        phase15_match = None
    if (
        not track_match
        and not phase4_match
        and not phase5_match
        and not (phase6_match and in_phase6)
        and not (phase7_match and in_phase7)
        and not (phase8_match and in_phase8)
        and not (phase9_match and in_phase9)
        and not (phase10_match and in_phase10)
        and not (phase11_match and in_phase11)
        and not (phase12_match and in_phase12)
        and not (phase13_match and in_phase13)
        and not (phase14_match and in_phase14)
        and not (phase15_match and in_phase15)
        and not (phase16_match and in_phase16)
    ):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    rel = path.relative_to(ROOT).as_posix()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    if track_match:
        track_id = int(track_match.group(1))
        key = f"track_{track_id}"
        metrics = _summary_metrics(data)
        return {
            "key": key,
            "kind": "adaptive_stop",
            "track": track_id,
            "experiment_id": data.get("experiment_id") or track_id,
            "title": data.get("title") or f"实验 {track_id}",
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            **metrics,
        }

    if phase4_match:
        exp_id = phase4_match.group(1)
        key = exp_id
        return {
            "key": key,
            "kind": "phase4",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": None,
            "timing_acc": None,
            "feasible": None,
            "strategy": None,
            "payload_keys": list(data.keys())[:12],
        }

    if phase5_match:
        exp_id = phase5_match.group(1)
        key = exp_id
        return {
            "key": key,
            "kind": "phase5",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": None,
            "timing_acc": None,
            "feasible": None,
            "strategy": None,
            "payload_keys": list(data.keys())[:12],
        }

    if phase10_match and in_phase10:
        exp_id = phase10_match.group(1)
        return {
            "key": exp_id,
            "kind": "phase10",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (data.get("test") or {}).get("accuracy"),
            "timing_acc": (data.get("test") or {}).get("stop_timing_acc"),
            "feasible": data.get("feasible"),
            "strategy": (data.get("test") or {}).get("strategy"),
            "payload_keys": list(data.keys())[:12],
        }

    if phase11_match and in_phase11:
        exp_id = phase11_match.group(1)
        test = data.get("test") or (data.get("full_419") or {}).get("test")
        return {
            "key": exp_id,
            "kind": "phase11",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (test or {}).get("accuracy"),
            "timing_acc": (test or {}).get("stop_timing_acc"),
            "feasible": data.get("feasible") or (data.get("proof") or {}).get("fully_proven"),
            "strategy": (test or {}).get("strategy"),
            "payload_keys": list(data.keys())[:12],
        }

    if phase12_match and in_phase12:
        exp_id = phase12_match.group(1)
        test = data.get("test") or (data.get("full_419") or {}).get("test")
        return {
            "key": exp_id,
            "kind": "phase12",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (test or {}).get("accuracy"),
            "timing_acc": (test or {}).get("stop_timing_acc"),
            "feasible": data.get("feasible") or data.get("fully_proven"),
            "strategy": (test or {}).get("strategy"),
            "payload_keys": list(data.keys())[:12],
        }

    if phase13_match and in_phase13:
        exp_id = phase13_match.group(1)
        test = data.get("test") or (data.get("full_419") or {}).get("test")
        return {
            "key": exp_id,
            "kind": "phase13",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (test or {}).get("accuracy"),
            "timing_acc": (test or {}).get("stop_timing_acc"),
            "feasible": data.get("feasible") or data.get("fully_proven"),
            "strategy": (test or {}).get("strategy"),
            "payload_keys": list(data.keys())[:12],
        }

    if phase14_match and in_phase14:
        exp_id = phase14_match.group(1)
        test = data.get("test") or (data.get("full_419_m2") or data.get("full_419") or {}).get("test")
        return {
            "key": exp_id,
            "kind": "phase14",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (test or {}).get("accuracy"),
            "timing_acc": (test or {}).get("stop_timing_acc"),
            "feasible": data.get("feasible") or data.get("strict_feasible") or data.get("deployable_mvp"),
            "strategy": (test or {}).get("strategy"),
            "payload_keys": list(data.keys())[:12],
        }

    if phase15_match and in_phase15:
        exp_id = phase15_match.group(1)
        test = data.get("test") or data.get("full_419") or data.get("test_40pct")
        return {
            "key": exp_id,
            "kind": "phase15",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (test or {}).get("accuracy") if isinstance(test, dict) else None,
            "timing_acc": (test or {}).get("stop_timing_acc") if isinstance(test, dict) else None,
            "feasible": data.get("feasible") or data.get("strict_feasible") or data.get("deployable_mvp"),
            "strategy": (test or {}).get("strategy") if isinstance(test, dict) else None,
            "payload_keys": list(data.keys())[:12],
        }

    if phase16_match and in_phase16:
        exp_id = phase16_match.group(1)
        test = data.get("full_419") or data.get("test")
        if not test and data.get("variants"):
            v = data["variants"].get("best_deployable_mvp") or {}
            test = v.get("full_419")
        return {
            "key": exp_id,
            "kind": "phase16",
            "track": None,
            "experiment_id": data.get("experiment_id") or exp_id,
            "title": data.get("title") or exp_id,
            "device": data.get("device") or "cuda",
            "sample_count": data.get("sample_count"),
            "duration_sec": data.get("duration_sec"),
            "finished_at": data.get("finished_at"),
            "source_file": rel,
            "source_mtime": mtime,
            "accuracy": (test or {}).get("accuracy") if isinstance(test, dict) else None,
            "timing_acc": (test or {}).get("stop_timing_acc") if isinstance(test, dict) else None,
            "feasible": data.get("feasible") or data.get("strict_feasible") or data.get("deployable_mvp"),
            "strategy": (test or {}).get("strategy") if isinstance(test, dict) else None,
            "payload_keys": list(data.keys())[:12],
        }

    phase_match = phase9_match or phase8_match or phase7_match or phase6_match
    exp_id = phase_match.group(1)
    key = exp_id
    if in_phase9:
        kind = "phase9"
    elif in_phase8:
        kind = "phase8"
    elif in_phase7:
        kind = "phase7"
    else:
        kind = "phase6"
    return {
        "key": key,
        "kind": kind,
        "track": None,
        "experiment_id": data.get("experiment_id") or exp_id,
        "title": data.get("title") or exp_id,
        "device": data.get("device") or "cuda",
        "sample_count": data.get("sample_count"),
        "duration_sec": data.get("duration_sec"),
        "finished_at": data.get("finished_at"),
        "source_file": rel,
        "source_mtime": mtime,
        "accuracy": None,
        "timing_acc": None,
        "feasible": None,
        "strategy": None,
        "payload_keys": list(data.keys())[:12],
    }


def collect_gpu_entries() -> list[dict[str, Any]]:
    """Newest file wins per key."""
    best: dict[str, tuple[float, dict]] = {}
    scan_roots = [
        FROM_A800,
        OUTBOX_RESULTS,
        *[ROOT / "results" / f"phase{n}" for n in range(6, 39)],
        PHASE4_CPU,
        PHASE5_CPU,
    ]
    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*_latest.json"):
            entry = _entry_from_json(path)
            if not entry:
                continue
            key = entry["key"]
            mtime = path.stat().st_mtime
            if key not in best or mtime > best[key][0]:
                best[key] = (mtime, entry)

    entries = [v[1] for v in best.values()]
    entries.sort(
        key=lambda e: (
            e.get("kind") or "",
            e.get("phase") or 999,
            e.get("track") or 999,
            e.get("experiment_id") or "",
        )
    )
    return entries


def build_manifest() -> dict[str, Any]:
    entries = collect_gpu_entries()
    highlights = [e for e in entries if e.get("feasible") is True]
    return {
        "ok": True,
        "updated_at": utc_now(),
        "host": "A800 · 115.190.90.101",
        "entry_count": len(entries),
        "feasible_count": len(highlights),
        "entries": entries,
        "highlights": [
            {
                "track": h.get("track"),
                "kind": h.get("kind"),
                "phase": h.get("phase"),
                "experiment_id": h.get("experiment_id"),
                "title": h.get("title"),
                "accuracy": h.get("accuracy"),
                "timing_acc": h.get("timing_acc"),
            }
            for h in highlights
        ],
    }


def gpu_by_track() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for e in collect_gpu_entries():
        if e.get("kind") == "adaptive_stop" and e.get("track") is not None:
            out[int(e["track"])] = e
    return out


def write_manifest(path: Path = OUT_PATH) -> dict:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = write_manifest()
    print(f"Wrote {OUT_PATH} ({manifest['entry_count']} entries, {manifest['feasible_count']} feasible)")


if __name__ == "__main__":
    main()
