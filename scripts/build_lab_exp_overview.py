#!/usr/bin/env python3
"""Build unified lab experiment overview — single numbered list, deduplicated."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_gpu_results_manifest import collect_gpu_entries  # noqa: E402

STATUS_FILE = ROOT / "results" / "status.json"
RESULTS_FILE = ROOT / "results" / "latest.json"
COMPARE_STATUS = ROOT / "results" / "compare_status.json"
COMPARE_RESULTS = ROOT / "results" / "compare_latest.json"
COMPARE_DEEP_STATUS = ROOT / "results" / "compare_deep_status.json"
COMPARE_DEEP_RESULTS = ROOT / "results" / "compare_deep_latest.json"
COMPARE_VARIANT_STATUS = ROOT / "results" / "compare_variant_status.json"
COMPARE_VARIANT_RESULTS = ROOT / "results" / "compare_variant_latest.json"
COMPARE_PATTERN_STATUS = ROOT / "results" / "compare_pattern_status.json"
COMPARE_PATTERN_RESULTS = ROOT / "results" / "compare_pattern_latest.json"
COMPARE_PUSH_STATUS = ROOT / "results" / "compare_boundary_push_status.json"
COMPARE_PUSH_RESULTS = ROOT / "results" / "compare_boundary_push_latest.json"
PERTURB_STATUS = ROOT / "results" / "model_perturb_status.json"
PERTURB_RESULTS = ROOT / "results" / "model_perturb_latest.json"
PUSH_DEEP_STATUS = ROOT / "results" / "compare_boundary_push_deep_status.json"
PUSH_DEEP_RESULTS = ROOT / "results" / "compare_boundary_push_deep_latest.json"
FEEDBACK_STATUS = ROOT / "results" / "feedback_schedule_status.json"
FEEDBACK_RESULTS = ROOT / "results" / "feedback_schedule_latest.json"
AUTO_SUBMIT_STATUS = ROOT / "results" / "auto_submit_status.json"
AUTO_SUBMIT_RESULTS = ROOT / "results" / "auto_submit_latest.json"

EARLY_EXP_META = [
    {"id": 1, "category": "A", "title": "全量 ProsQA 扫描", "anchor": "exp1Panel", "status": STATUS_FILE, "results": RESULTS_FILE},
    {"id": 2, "category": "A", "title": "多种数据对比", "anchor": "exp2Panel", "status": COMPARE_STATUS, "results": COMPARE_RESULTS},
    {"id": 3, "category": "A", "title": "深边界探测（5–6 步）", "anchor": "exp3Panel", "status": COMPARE_DEEP_STATUS, "results": COMPARE_DEEP_RESULTS},
    {"id": 4, "category": "A", "title": "构造 × 监督对照", "anchor": "exp4Panel", "status": COMPARE_VARIANT_STATUS, "results": COMPARE_VARIANT_RESULTS},
    {"id": 5, "category": "A", "title": "规律寻探", "anchor": "exp5Panel", "status": COMPARE_PATTERN_STATUS, "results": COMPARE_PATTERN_RESULTS},
    {"id": 6, "category": "B", "title": "边界上推（同质延长）", "anchor": "exp6Panel", "status": COMPARE_PUSH_STATUS, "results": COMPARE_PUSH_RESULTS},
    {"id": 7, "category": "B", "title": "只改模型数值", "anchor": "exp7Panel", "status": PERTURB_STATUS, "results": PERTURB_RESULTS},
    {"id": 8, "category": "B", "title": "7–8 步深边界上推", "anchor": "exp8Panel", "status": PUSH_DEEP_STATUS, "results": PUSH_DEEP_RESULTS},
    {"id": 9, "category": "C", "title": "4 步以后如何不跌？", "anchor": "exp9Panel", "status": FEEDBACK_STATUS, "results": FEEDBACK_RESULTS},
    {"id": 10, "category": "C", "title": "无标签自动配参", "anchor": "exp10Panel", "status": AUTO_SUBMIT_STATUS, "results": AUTO_SUBMIT_RESULTS},
]

ADAPTIVE_STOP_OVERVIEW_META = [
    {"id": 11, "category": "D", "title": "训练 stop head", "anchor": "exp11Panel", "slug": ""},
    {"id": 12, "category": "D", "title": "改进 stop head", "anchor": "exp12Panel", "slug": "v2"},
    {"id": 13, "category": "D", "title": "Rich stop head", "anchor": "exp13Panel", "slug": "v3"},
    {"id": 14, "category": "D", "title": "is_correct + 平衡校准", "anchor": "exp14Panel", "slug": "v4"},
    {"id": 15, "category": "D", "title": "Coconut 联合微调", "anchor": "exp15Panel", "slug": "v5"},
    {"id": 16, "category": "E", "title": "结构 + 收敛", "anchor": "exp16Panel", "slug": "v16", "direction": "结构通解+收敛"},
    {"id": 17, "category": "E", "title": "BFS + Δ 修正", "anchor": "exp17Panel", "slug": "v17", "direction": "结构+残差"},
    {"id": 18, "category": "E", "title": "first_correct + timing", "anchor": "exp18Panel", "slug": "v18", "direction": "直接优化 timing"},
    {"id": 19, "category": "E", "title": "收敛 OR 稳定", "anchor": "exp19Panel", "slug": "v19", "direction": "OR 非 AND"},
    {"id": 20, "category": "E", "title": "BFS + v4 头", "anchor": "exp20Panel", "slug": "v20", "direction": "路由+学习组合"},
    {"id": 21, "category": "D", "title": "BFS + v5 联合", "anchor": "exp21Panel", "slug": "v21", "direction": "Exp15后续"},
    {"id": 22, "category": "E", "title": "v4 + timing 校准", "anchor": "exp22Panel", "slug": "v22", "direction": "Exp20·timing"},
    {"id": 23, "category": "E", "title": "v4 + route 上界", "anchor": "exp23Panel", "slug": "v23", "direction": "acc·timing折中"},
    {"id": 24, "category": "E", "title": "first_correct 门控", "anchor": "exp24Panel", "slug": "v24", "direction": "结构+首次答对"},
    {"id": 25, "category": "E", "title": "first_correct + cap", "anchor": "exp25Panel", "slug": "v25", "direction": "Exp24·cap d"},
    {"id": 26, "category": "E", "title": "分跳数诊断", "anchor": "exp26Panel", "slug": "v26", "direction": "按跳数拆分"},
    {"id": 27, "category": "E", "title": "4跳 d+1 cap", "anchor": "exp27Panel", "slug": "v27", "direction": "分跳数 cap"},
    {"id": 28, "category": "E", "title": "soft floor fc", "anchor": "exp28Panel", "slug": "v28", "direction": "放宽下界"},
    {"id": 29, "category": "E", "title": "分跳 soft/strict", "anchor": "exp29Panel", "slug": "v29", "direction": "4跳 soft"},
    {"id": 30, "category": "E", "title": "3跳fc+4跳v4", "anchor": "exp30Panel", "slug": "v30", "direction": "分跳混合"},
    {"id": 31, "category": "F", "title": "结构预算·单次", "anchor": "exp31Panel", "slug": "v31", "direction": "通解A·不试步"},
    {"id": 32, "category": "F", "title": "图MLP预算", "anchor": "exp32Panel", "slug": "v32", "direction": "通解B·学习"},
    {"id": 33, "category": "F", "title": "图Δ残差", "anchor": "exp33Panel", "slug": "v33", "direction": "结构+残差"},
    {"id": 34, "category": "F", "title": "完美预算上界", "anchor": "exp34Panel", "slug": "v34", "direction": "单次上界"},
    {"id": 35, "category": "F", "title": "上界·seed修正", "anchor": "exp35Panel", "slug": "v35", "direction": "上界修正"},
    {"id": 36, "category": "F", "title": "train查表", "anchor": "exp36Panel", "slug": "v36", "direction": "可部署查表"},
    {"id": 37, "category": "F", "title": "4跳→3步", "anchor": "exp37Panel", "slug": "v37", "direction": "规则·单次"},
    {"id": 38, "category": "F", "title": "4跳→d-1", "anchor": "exp38Panel", "slug": "v38", "direction": "规则·单次"},
    {"id": 39, "category": "F", "title": "d4二分类MLP", "anchor": "exp39Panel", "slug": "v39", "direction": "4跳学习"},
    {"id": 40, "category": "F", "title": "(d,asym)查表", "anchor": "exp40Panel", "slug": "v40", "direction": "细查表"},
    {"id": 41, "category": "F", "title": "不对称→3", "anchor": "exp41Panel", "slug": "v41", "direction": "不对称规则"},
    {"id": 42, "category": "F", "title": "d4完美上界", "anchor": "exp42Panel", "slug": "v42", "direction": "d4上界"},
    {"id": 43, "category": "F", "title": "d4阈值校准", "anchor": "exp43Panel", "slug": "v43", "direction": "d4阈值"},
    {"id": 44, "category": "F", "title": "d4三分类", "anchor": "exp44Panel", "slug": "v44", "direction": "d4三分类"},
    {"id": 45, "category": "F", "title": "d3二分类", "anchor": "exp45Panel", "slug": "v45", "direction": "d3专项"},
    {"id": 46, "category": "F", "title": "d4 kNN", "anchor": "exp46Panel", "slug": "v46", "direction": "d4 kNN"},
    {"id": 47, "category": "F", "title": "前缀预算", "anchor": "exp47Panel", "slug": "v47", "direction": "模型自预测"},
    {"id": 48, "category": "F", "title": "前缀Δ", "anchor": "exp48Panel", "slug": "v48", "direction": "模型自预测Δ"},
    {"id": 49, "category": "F", "title": "前缀阈·标签", "anchor": "exp49Panel", "slug": "v49", "direction": "阈值校准"},
    {"id": 50, "category": "F", "title": "前缀阈·acc", "anchor": "exp50Panel", "slug": "v50", "direction": "acc校准"},
    {"id": 51, "category": "F", "title": "前缀+kNN", "anchor": "exp51Panel", "slug": "v51", "direction": "集成"},
    {"id": 52, "category": "F", "title": "前缀保守", "anchor": "exp52Panel", "slug": "v52", "direction": "保守回退"},
    {"id": 53, "category": "G", "title": "在线 head∨stable", "anchor": "exp53Panel", "slug": "v53", "direction": "在线自停"},
    {"id": 54, "category": "G", "title": "在线 head∨conv", "anchor": "exp54Panel", "slug": "v54", "direction": "在线自停"},
    {"id": 55, "category": "G", "title": "在线三重OR", "anchor": "exp55Panel", "slug": "v55", "direction": "在线自停"},
]

PHASE_ROLLUP_PRIORITY = (
    "final_rollup",
    "proof_rollup",
    "absolute_final",
    "project_complete",
    "project_locked",
    "deploy_spec",
    "final_manifest",
    "champion_validate",
    "project_closure",
)

PHASE_GPU_FIRST_ID = 56


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _running_from_status(status: dict, has_summary: bool) -> bool:
    running = bool(status.get("running")) and status.get("phase") not in ("done", "error", None)
    if status.get("phase") == "done" or has_summary:
        return False
    return running


def _early_accuracy(exp_id: int, results: dict) -> Optional[float]:
    if not results or not results.get("ok"):
        return None
    if exp_id == 1:
        b = results.get("boundary") or {}
        return b.get("max_accuracy")
    best = None
    for row in results.get("rows") or []:
        v = row.get("max_accuracy")
        if v is not None and (best is None or v > best):
            best = v
    if best is not None:
        return best
    for sl in results.get("slices") or []:
        for pt in sl.get("latent_sweep") or []:
            v = pt.get("accuracy")
            if v is not None and (best is None or v > best):
                best = v
    return best


def _adaptive_stop_paths(exp_id: int) -> tuple[Path, Path]:
    meta = next(m for m in ADAPTIVE_STOP_OVERVIEW_META if m["id"] == exp_id)
    slug = meta["slug"]
    if slug == "":
        base = "adaptive_stop"
    elif slug == "v2":
        base = "adaptive_stop_v2"
    else:
        base = f"adaptive_stop_{slug}"
    return (
        ROOT / "results" / f"{base}_status.json",
        ROOT / "results" / f"{base}_latest.json",
    )


def _adaptive_summary_metrics(summary: dict) -> tuple[Optional[float], Optional[float], Optional[bool]]:
    acc = (
        summary.get("main_strategy_accuracy")
        or summary.get("joint_correctness_stop_accuracy")
        or summary.get("correctness_stop_accuracy")
        or summary.get("rich_stop_accuracy")
        or summary.get("trained_stop_v2_accuracy")
        or summary.get("trained_stop_accuracy")
    )
    timing = (
        summary.get("main_strategy_timing_acc")
        or summary.get("joint_correctness_stop_timing_acc")
        or summary.get("correctness_stop_timing_acc")
        or summary.get("rich_stop_timing_acc")
        or summary.get("trained_stop_v2_timing_acc")
        or summary.get("trained_stop_timing_acc")
    )
    feasible = summary.get("trainable_stop_feasible")
    return acc, timing, feasible


def _gpu_panel_anchor(key: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", key)
    return f"gpuPanel_{safe}"


def _merge_metrics(
    local_acc, local_timing, local_feasible,
    remote_acc, remote_timing, remote_feasible,
) -> tuple[Optional[float], Optional[float], Optional[bool]]:
    """Prefer A800 validation metrics when both local and remote exist."""
    acc = remote_acc if remote_acc is not None else local_acc
    timing = remote_timing if remote_timing is not None else local_timing
    if remote_feasible is not None:
        feasible = remote_feasible
    else:
        feasible = local_feasible
    return acc, timing, feasible


def _canonical_phase_score(key: str) -> int:
    k = (key or "").lower()
    for i, pat in enumerate(PHASE_ROLLUP_PRIORITY):
        if pat in k:
            return 1000 - i
    return 0


def _pick_canonical_phase_entries(gpu_batches: list[dict]) -> dict[int, dict]:
    by_phase: dict[int, list[dict]] = {}
    for e in gpu_batches:
        phase = e.get("phase")
        if phase is None:
            continue
        by_phase.setdefault(int(phase), []).append(e)
    out: dict[int, dict] = {}
    for phase, entries in by_phase.items():
        out[phase] = max(
            entries,
            key=lambda e: (_canonical_phase_score(e.get("key") or ""), e.get("source_mtime") or ""),
        )
    return out


def _row_base(
    *,
    exp_id: int,
    category: str,
    key: str,
    anchor: Optional[str],
    title: str,
    direction: Optional[str] = None,
    running: bool = False,
    phase: str = "idle",
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "row_type": "numbered",
        "category": category,
        "exp_id": exp_id,
        "key": key,
        "anchor": anchor,
        "title": title,
        "direction": direction,
        "running": running,
        "phase": phase,
        "error": error,
        "accuracy": None,
        "timing": None,
        "feasible": None,
        "source_file": None,
        "insight": None,
        "duration_sec": None,
        "gpu_detail": False,
        "gpu_phase": None,
    }


def build_lab_exp_overview() -> dict[str, Any]:
    gpu_entries = collect_gpu_entries()
    gpu_by_track: dict[int, dict] = {}
    gpu_batches: list[dict] = []
    for e in gpu_entries:
        if e.get("kind") == "adaptive_stop" and e.get("track") is not None:
            gpu_by_track[int(e["track"])] = e
        else:
            gpu_batches.append(e)

    canonical_phases = _pick_canonical_phase_entries(gpu_batches)

    rows: list[dict[str, Any]] = []
    running_count = 0

    for meta in EARLY_EXP_META:
        status = _read_json(meta["status"], {})
        results = _read_json(meta["results"], {})
        summary = status.get("summary") or {}
        has_data = bool(results.get("ok")) or bool(summary)
        running = _running_from_status(status, bool(summary))
        if running:
            running_count += 1
        row = _row_base(
            exp_id=meta["id"],
            category=meta["category"],
            key=f"exp_{meta['id']}",
            anchor=meta["anchor"],
            title=meta["title"],
            running=running,
            phase=status.get("phase") or ("done" if has_data else "idle"),
            error=status.get("error") or results.get("error"),
        )
        local_acc = _early_accuracy(meta["id"], results)
        row["accuracy"], row["timing"], row["feasible"] = _merge_metrics(
            local_acc, None, None, None, None, None,
        )
        row["duration_sec"] = results.get("duration_sec") or status.get("duration_sec")
        rows.append(row)

    for meta in ADAPTIVE_STOP_OVERVIEW_META:
        status_path, results_path = _adaptive_stop_paths(meta["id"])
        status = _read_json(status_path, {})
        results = _read_json(results_path, {})
        summary = status.get("summary") or results.get("summary") or {}
        running = _running_from_status(status, bool(summary))
        if running:
            running_count += 1
        local_acc, local_timing, local_feasible = _adaptive_summary_metrics(summary)
        gpu = gpu_by_track.get(meta["id"])
        row = _row_base(
            exp_id=meta["id"],
            category=meta["category"],
            key=f"exp_{meta['id']}",
            anchor=meta["anchor"],
            title=meta["title"],
            direction=meta.get("direction"),
            running=running,
            phase=status.get("phase") or ("done" if summary else "idle"),
            error=status.get("error") or results.get("error"),
        )
        row["accuracy"], row["timing"], row["feasible"] = _merge_metrics(
            local_acc, local_timing, local_feasible,
            (gpu or {}).get("accuracy"),
            (gpu or {}).get("timing_acc"),
            (gpu or {}).get("feasible"),
        )
        row["duration_sec"] = results.get("duration_sec") or status.get("duration_sec")
        if gpu:
            row["source_file"] = gpu.get("source_file")
        rows.append(row)

    next_id = PHASE_GPU_FIRST_ID
    for phase in sorted(canonical_phases.keys()):
        if phase < 4:
            continue
        e = canonical_phases[phase]
        key = e.get("key") or f"phase{phase}_rollup"
        row = _row_base(
            exp_id=next_id,
            category=f"P{phase}",
            key=key,
            anchor=_gpu_panel_anchor(key),
            title=e.get("title") or e.get("experiment_id") or key,
            direction=f"Phase {phase}",
            phase="done" if e.get("accuracy") is not None else "idle",
        )
        row["accuracy"] = e.get("accuracy")
        row["timing"] = e.get("timing_acc")
        row["feasible"] = e.get("feasible")
        row["source_file"] = e.get("source_file")
        row["insight"] = e.get("insight")
        row["duration_sec"] = e.get("duration_sec")
        row["gpu_detail"] = True
        row["gpu_phase"] = phase
        rows.append(row)
        next_id += 1

    rows.sort(key=lambda r: r.get("exp_id") or 0)

    feasible_count = sum(1 for r in rows if r.get("feasible") is True)
    with_detail = sum(1 for r in rows if r.get("gpu_detail"))

    return {
        "ok": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "running_count": running_count,
        "row_count": len(rows),
        "numbered_count": len(rows),
        "phase_gpu_count": with_detail,
        "feasible_count": feasible_count,
        "rows": rows,
    }


if __name__ == "__main__":
    out = build_lab_exp_overview()
    out_path = ROOT / "results" / "lab_exp_overview.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"rows={out['row_count']} numbered={out['numbered_count']} "
        f"phase_detail={out['phase_gpu_count']}"
    )
