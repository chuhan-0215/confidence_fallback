#!/usr/bin/env python3
"""多数据集对比实验 — 在同一 Coconut 模型上扫描多种 ProsQA 子集边界。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from dataset_registry import (  # noqa: E402
    default_boundary_push_deep_slice_ids,
    default_boundary_push_slice_ids,
    default_compare_slice_ids,
    default_deep_compare_slice_ids,
    default_pattern_compare_slice_ids,
    default_variant_compare_slice_ids,
    list_slices,
    load_master,
    load_slice,
)
from eval_profile import parse_eval_profile  # noqa: E402
from evaluate_coconut import dataset_meta, load_coconut_model, resolve_device  # noqa: E402
from run_experiment import ensure_checkpoint, run_on_dataset, sweep_latent_steps, write_status  # noqa: E402


def build_comparison(slice_results: List[dict]) -> dict:
    table = []
    for sr in slice_results:
        boundary = sr.get("boundary") or {}
        ds = sr.get("dataset") or {}
        mean_hops = ds.get("mean_reasoning_hops")
        acc_at_depth = None
        if mean_hops is not None:
            sweep_map = {r["n_latent"]: r["accuracy"] for r in (sr.get("latent_sweep") or [])}
            acc_at_depth = sweep_map.get(int(round(float(mean_hops))))
        table.append(
            {
                "id": sr.get("slice_id"),
                "label": sr.get("slice_label"),
                "description": sr.get("slice_description"),
                "count": ds.get("count"),
                "mean_reasoning_hops": mean_hops,
                "mean_diameter": ds.get("mean_diameter_from_root"),
                "boundary": boundary.get("recommended_latent_steps"),
                "max_accuracy": boundary.get("max_accuracy"),
                "max_accuracy_at_steps": boundary.get("max_accuracy_at_steps"),
                "acc_at_depth": acc_at_depth,
                "duration_sec": sr.get("duration_sec"),
                "construction": (sr.get("slice_meta") or {}).get("construction"),
                "supervision": (sr.get("slice_meta") or {}).get("supervision_label"),
                "eval_profile": sr.get("eval_profile"),
            }
        )

    boundaries = sorted({t["boundary"] for t in table if t["boundary"] is not None})
    insights: List[str] = []

    if len(boundaries) >= 2:
        insights.append(
            f"不同子集检测到的推荐边界分布在 {boundaries[0]}–{boundaries[-1]} 步之间，"
            "说明「换数据边界会变」——边界取决于样本结构与推理跳数分布。"
        )
    elif len(boundaries) == 1:
        insights.append(
            f"当前各子集推荐边界均为 {boundaries[0]} 步；若扩大样本量或换随机种子，边界仍可能波动。"
        )

    by_id = {t["id"]: t for t in table}
    h3, h4 = by_id.get("hops_3"), by_id.get("hops_4")
    if h3 and h4 and h3["boundary"] is not None and h4["boundary"] is not None:
        if h3["boundary"] == h4["boundary"]:
            insights.append(
                f"3 跳与 4 跳子集边界同为 {h3['boundary']} 步，"
                f"但最高准确率分别为 {(h3['max_accuracy'] or 0) * 100:.1f}% 与 {(h4['max_accuracy'] or 0) * 100:.1f}%。"
            )
        else:
            insights.append(
                f"3 跳子集边界 {h3['boundary']} 步（acc {(h3['max_accuracy'] or 0) * 100:.1f}%），"
                f"4 跳子集边界 {h4['boundary']} 步（acc {(h4['max_accuracy'] or 0) * 100:.1f}%）。"
            )

    ra, rb = by_id.get("random_a"), by_id.get("random_b")
    if ra and rb and ra["boundary"] != rb["boundary"]:
        insights.append(
            f"两组随机子集边界不同（A={ra['boundary']} 步，B={rb['boundary']} 步），"
            "说明小样本随机划分也会改变检测到的边界。"
        )

    d3, d4 = by_id.get("diameter_3"), by_id.get("diameter_4")
    if d3 and d4 and d3["boundary"] is not None and d4["boundary"] is not None:
        insights.append(
            f"图直径 3 子集边界 {d3['boundary']} 步，直径 4 子集边界 {d4['boundary']} 步。"
        )

    s5, s6 = by_id.get("syn_chain_5"), by_id.get("syn_chain_6")
    if s5 and s6 and s5["boundary"] is not None and s6["boundary"] is not None:
        insights.append(
            f"合成 5 跳链边界 {s5['boundary']} 步，合成 6 跳链边界 {s6['boundary']} 步。"
        )
        if s6["boundary"] >= 5:
            insights.append("6 跳合成任务边界 ≥5 步，说明任务深度足够时边界可上移到 5–6 步。")
        elif s5["boundary"] >= 5:
            insights.append("5 跳合成任务边界 ≥5 步。")

    pw = by_id.get("prosqa_diameter_wide")
    if pw and s6 and pw["boundary"] is not None and s6["boundary"] is not None:
        if s6["boundary"] > pw["boundary"]:
            insights.append(
                f"合成 6 跳边界（{s6['boundary']} 步）高于 ProsQA 宽图（{pw['boundary']} 步），"
                "说明原数据边界偏低主因是任务深度不足，而非模型硬上限。"
            )

    # 变体实验：构造 vs 监督
    v7 = by_id.get("v_chain_7_dense")
    v6 = by_id.get("v_chain_6_dense")
    if v7 and v6 and v7.get("boundary") is not None and v6.get("boundary") is not None:
        insights.append(
            f"稠密 7 跳边界 {v7['boundary']} 步 vs 6 跳 {v6['boundary']} 步。"
        )

    sym = by_id.get("v_chain_6_symbol")
    idx = by_id.get("v_chain_6_dense")
    if sym and idx and sym.get("boundary") is not None and idx.get("boundary") is not None:
        if sym["boundary"] != idx["boundary"]:
            insights.append(
                f"同一 6 跳数据：索引监督边界 {idx['boundary']} 步，符号监督 {sym['boundary']} 步。"
            )
        else:
            insights.append(
                f"索引与符号监督边界同为 {idx['boundary']} 步（6 跳稠密链）。"
            )

    fix = by_id.get("v_chain_6_fixed")
    if fix and idx and fix.get("boundary") is not None and idx.get("boundary") is not None:
        if fix["boundary"] != idx["boundary"]:
            insights.append(
                f"固定边序边界 {fix['boundary']} 步 vs shuffle 边序 {idx['boundary']} 步。"
            )

    ext6 = by_id.get("v_extend_6")
    if ext6 and v6 and ext6.get("boundary") is not None and v6.get("boundary") is not None:
        insights.append(
            f"真实图延长 6 跳边界 {ext6['boundary']} 步 vs 人工稠密 6 跳 {v6['boundary']} 步。"
        )

    acc_range = [t["max_accuracy"] for t in table if t["max_accuracy"] is not None]

    v_ext5 = by_id.get("v_extend_5") or by_id.get("push_ext5_mixed")
    v_ext6 = by_id.get("v_extend_6") or by_id.get("push_ext6_mixed")
    if v_ext5 and v_ext6:
        b5, b6 = v_ext5.get("boundary"), v_ext6.get("boundary")
        if b5 is not None and b6 is not None:
            insights.append(
                f"真实图延长：5 跳边界 {b5} 步（acc {(v_ext5['max_accuracy'] or 0) * 100:.1f}%），"
                f"6 跳边界 {b6} 步（acc {(v_ext6['max_accuracy'] or 0) * 100:.1f}%）。"
            )

    p54 = by_id.get("push_ext5_from4")
    p64 = by_id.get("push_ext6_from4")
    p53 = by_id.get("push_ext5_from3")
    p63 = by_id.get("push_ext6_from3")
    if p54 and p64 and p54.get("boundary") is not None and p64.get("boundary") is not None:
        insights.append(
            f"同质 4 跳基线延长：5 跳边界 {p54['boundary']} 步，6 跳边界 {p64['boundary']} 步。"
        )
    if p53 and p63 and p53.get("boundary") is not None and p63.get("boundary") is not None:
        insights.append(
            f"同质 3 跳基线延长：5 跳边界 {p53['boundary']} 步，6 跳边界 {p63['boundary']} 步。"
        )
    if p64 and p63 and p64.get("boundary") is not None and p63.get("boundary") is not None:
        if p63["boundary"] == 5 and p64["boundary"] > 6:
            insights.append(
                "5 跳边界：从 3 跳基线延长可对齐到 5 步；从 4 跳基线延长易过冲（边界偏高）。"
            )
        elif p64["boundary"] >= p63["boundary"]:
            insights.append(
                f"4 跳基线延长 6 跳边界（{p64['boundary']} 步）≥ 3 跳基线（{p63['boundary']} 步）。"
            )

    mix56 = by_id.get("push_mix_56_from4")
    if mix56 and mix56.get("boundary") is not None:
        insights.append(
            f"4 跳基线 5/6 跳混合集边界 {mix56['boundary']} 步（acc {(mix56['max_accuracy'] or 0) * 100:.1f}%）。"
        )

    for sid, label in (
        ("push_ext7_from3", "7 跳同质延长"),
        ("push_ext8_from3", "8 跳同质延长"),
        ("push_ext7_mixed", "7 跳混合延长"),
    ):
        row = by_id.get(sid)
        if not row or row.get("boundary") is None:
            continue
        d = row.get("mean_reasoning_hops")
        acc_d = row.get("acc_at_depth")
        acc_s = f"，acc@{int(round(d))}={(acc_d or 0) * 100:.1f}%" if d and acc_d is not None else ""
        insights.append(
            f"{label}：报边界 {row['boundary']} 步，峰值 {(row['max_accuracy'] or 0) * 100:.1f}%{acc_s}。"
        )

    p73 = by_id.get("push_ext7_from3")
    p83 = by_id.get("push_ext8_from3")
    if p73 and p83 and p73.get("acc_at_depth") is not None and p83.get("acc_at_depth") is not None:
        if p73["acc_at_depth"] >= 0.9 and p73.get("boundary") == 7:
            insights.append("7 跳同质延长：边界与 acc@7 双高——上推公式在 d=7 仍成立。")
        elif p73["acc_at_depth"] < 0.85:
            insights.append(
                f"7 跳同质延长 acc@7 仅 {(p73['acc_at_depth'] or 0) * 100:.1f}%——"
                "仅改测试延长不够，需配合同深度训练。"
            )

    summary = " · ".join(insights[:5]) if insights else "完成多数据集扫描，见下表对比。"

    return {
        "summary": summary,
        "boundary_range": {"min": boundaries[0], "max": boundaries[-1]} if boundaries else None,
        "accuracy_range": {
            "min": round(min(acc_range), 4),
            "max": round(max(acc_range), 4),
        }
        if acc_range
        else None,
        "table": table,
        "insights": insights,
    }


def run_compare_experiment(
    slice_ids: Optional[List[str]] = None,
    max_samples_per_slice: Optional[int] = None,
    latent_min: int = 1,
    latent_max: int = 8,
    device_name: str = "auto",
    progress_cb: Optional[Callable] = None,
    status_file: Optional[str] = None,
    experiment_type: str = "multi_dataset_compare",
) -> dict:
    config_path = ROOT / "configs" / "symbol-2layer-8head-768dim.json"
    checkpoint = ROOT / "checkpoints" / "checkpoint_300"
    device = resolve_device(device_name)
    checkpoint = ensure_checkpoint(checkpoint)

    is_deep = experiment_type == "deep_boundary_compare"
    is_variant = experiment_type == "variant_boundary_compare"
    is_pattern = experiment_type == "pattern_boundary_compare"
    is_push = experiment_type == "boundary_push_compare"
    is_push_deep = experiment_type == "boundary_push_deep_compare"
    if slice_ids:
        ids = slice_ids
    elif is_push_deep:
        ids = default_boundary_push_deep_slice_ids()
    elif is_push:
        ids = default_boundary_push_slice_ids()
    elif is_pattern:
        ids = default_pattern_compare_slice_ids()
    elif is_variant:
        ids = default_variant_compare_slice_ids()
    elif is_deep:
        ids = default_deep_compare_slice_ids()
    else:
        ids = default_compare_slice_ids()
    master = load_master()
    slice_manifest = list_slices(
        deep=is_deep,
        variant=is_variant,
        pattern=is_pattern,
        push=is_push,
        push_deep=is_push_deep,
    )

    if status_file:
        write_status(
            Path(status_file),
            {
                "running": True,
                "phase": "loading_model",
                "slice_total": len(ids),
                "slice_done": 0,
                "slices": slice_manifest,
            },
        )

    t0 = time.time()
    model, tokenizer = load_coconut_model(checkpoint, config_path, device)

    slice_results = []
    for si, slice_id in enumerate(ids):
        meta, dataset = load_slice(slice_id, max_samples=max_samples_per_slice, master=master)
        if not dataset:
            continue
        eval_profile = parse_eval_profile(meta.get("eval_profile"))

        def slice_progress(done, total, row, _sid=slice_id, _si=si):
            payload = {
                "running": True,
                "phase": "slice_sweep",
                "slice_id": _sid,
                "slice_index": _si,
                "slice_total": len(ids),
                "slice_done": _si,
                "progress": {"done": done, "total": total, "last": row},
            }
            if status_file:
                write_status(Path(status_file), payload)
            if progress_cb:
                progress_cb(_si, len(ids), _sid, done, total, row)

        slice_t0 = time.time()
        one = run_on_dataset(
            model,
            tokenizer,
            dataset,
            device,
            latent_min=latent_min,
            latent_max=latent_max,
            skip_hop_groups=True,
            progress_cb=slice_progress,
            slice_meta=meta,
            eval_profile=eval_profile,
        )
        one["duration_sec"] = round(time.time() - slice_t0, 2)
        one["slice_id"] = meta["slice_id"]
        one["slice_label"] = meta["label"]
        one["slice_description"] = meta["description"]
        slice_results.append(one)

        if status_file:
            write_status(
                Path(status_file),
                {
                    "running": True,
                    "phase": "slice_done",
                    "slice_id": slice_id,
                    "slice_index": si + 1,
                    "slice_total": len(ids),
                    "partial_results": slice_results,
                },
            )

    comparison = build_comparison(slice_results)
    result = {
        "ok": True,
        "experiment_type": experiment_type,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "params": {
            "slice_ids": ids,
            "max_samples_per_slice": max_samples_per_slice,
            "latent_min": latent_min,
            "latent_max": latent_max,
        },
        "slice_manifest": slice_manifest,
        "slices": slice_results,
        "comparison": comparison,
        "references": {
            "paper": "https://arxiv.org/abs/2505.12514",
            "code": "https://github.com/Ber666/reasoning-by-superposition",
        },
    }

    if status_file:
        write_status(
            Path(status_file),
            {
                "running": False,
                "phase": "done",
                "slice_total": len(ids),
                "slice_done": len(slice_results),
                "comparison": comparison,
            },
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset boundary comparison")
    parser.add_argument(
        "--slices",
        default="",
        help="Comma-separated slice ids (default: all registered slices)",
    )
    parser.add_argument(
        "--max-samples-per-slice",
        type=int,
        default=60,
        help="Cap per slice; use 0 for slice default caps only",
    )
    parser.add_argument("--latent-min", type=int, default=1)
    parser.add_argument("--latent-max", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deep", action="store_true", help="Run deep boundary slice set (5–6 hop)")
    parser.add_argument(
        "--variant",
        action="store_true",
        help="Run variant boundary compare (construction × supervision)",
    )
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="Run pattern-finding compare (mix ladder + hop×diameter)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Run boundary push compare (homogeneous extend to 5–6 hops)",
    )
    parser.add_argument(
        "--push-deep",
        action="store_true",
        help="Run boundary push deep compare (extend to 7–8 hops, latent 1–12)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON (default: compare_latest or compare_deep_latest)",
    )
    parser.add_argument("--status-file", default="")
    parser.add_argument("--list-slices", action="store_true")
    parser.add_argument("--list-deep-slices", action="store_true")
    parser.add_argument("--list-variant-slices", action="store_true")
    parser.add_argument("--list-pattern-slices", action="store_true")
    parser.add_argument("--list-push-slices", action="store_true")
    parser.add_argument("--list-push-deep-slices", action="store_true")
    args = parser.parse_args()

    if args.list_slices:
        print(json.dumps(list_slices(deep=False), ensure_ascii=False, indent=2))
        return
    if args.list_deep_slices:
        print(json.dumps(list_slices(deep=True), ensure_ascii=False, indent=2))
        return
    if args.list_variant_slices:
        print(json.dumps(list_slices(variant=True), ensure_ascii=False, indent=2))
        return
    if args.list_pattern_slices:
        print(json.dumps(list_slices(pattern=True), ensure_ascii=False, indent=2))
        return
    if args.list_push_slices:
        print(json.dumps(list_slices(push=True), ensure_ascii=False, indent=2))
        return
    if args.list_push_deep_slices:
        print(json.dumps(list_slices(push_deep=True), ensure_ascii=False, indent=2))
        return

    slice_ids = [s.strip() for s in args.slices.split(",") if s.strip()] or None
    cap = args.max_samples_per_slice if args.max_samples_per_slice > 0 else None
    if args.pattern:
        exp_type = "pattern_boundary_compare"
    elif args.push_deep:
        exp_type = "boundary_push_deep_compare"
    elif args.push:
        exp_type = "boundary_push_compare"
    elif args.variant:
        exp_type = "variant_boundary_compare"
    elif args.deep:
        exp_type = "deep_boundary_compare"
    else:
        exp_type = "multi_dataset_compare"
    latent_max = args.latent_max
    if args.deep and args.latent_max == 8:
        latent_max = 10
    if args.variant and args.latent_max == 8:
        latent_max = 10
    if args.push_deep and args.latent_max == 8:
        latent_max = 12
    if args.push and args.latent_max == 8:
        latent_max = 10

    if args.pattern:
        default_out = ROOT / "results" / "compare_pattern_latest.json"
        default_status = ROOT / "results" / "compare_pattern_status.json"
    elif args.push_deep:
        default_out = ROOT / "results" / "compare_boundary_push_deep_latest.json"
        default_status = ROOT / "results" / "compare_boundary_push_deep_status.json"
    elif args.push:
        default_out = ROOT / "results" / "compare_boundary_push_latest.json"
        default_status = ROOT / "results" / "compare_boundary_push_status.json"
    elif args.variant:
        default_out = ROOT / "results" / "compare_variant_latest.json"
        default_status = ROOT / "results" / "compare_variant_status.json"
    elif args.deep:
        default_out = ROOT / "results" / "compare_deep_latest.json"
        default_status = ROOT / "results" / "compare_deep_status.json"
    else:
        default_out = ROOT / "results" / "compare_latest.json"
        default_status = ROOT / "results" / "compare_status.json"

    out = Path(args.output or default_out)
    status = Path(args.status_file or default_status)

    def progress(si, total, sid, done, step_total, row):
        print(f"[slice {si + 1}/{total} {sid}] [{done}/{step_total}]", row, flush=True)

    try:
        result = run_compare_experiment(
            slice_ids=slice_ids,
            max_samples_per_slice=cap,
            latent_min=args.latent_min,
            latent_max=latent_max,
            device_name=args.device,
            progress_cb=progress,
            status_file=str(status),
            experiment_type=exp_type,
        )
    except Exception as exc:
        sf = status
        write_status(sf, {"running": False, "phase": "error", "error": str(exc)})
        raise

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
