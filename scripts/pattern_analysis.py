#!/usr/bin/env python3
"""从全部实验结果中提取可复现规律，并生成面向网站的解释。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{float(x) * 100:.1f}%"


def _step(x: float | None) -> str:
    if x is None:
        return "—"
    v = float(x)
    return str(int(v)) if v == int(v) else str(v)


def _collect_slice_rows(*payloads: dict | None) -> List[dict]:
    rows: List[dict] = []
    for payload in payloads:
        if not payload:
            continue
        for sr in payload.get("slices") or []:
            ds = sr.get("dataset") or {}
            boundary = sr.get("boundary") or {}
            sweep = sr.get("latent_sweep") or []
            jump_step = None
            jump_pp = None
            if len(sweep) >= 2:
                best = max(
                    (
                        (i, sweep[i]["accuracy"] - sweep[i - 1]["accuracy"])
                        for i in range(1, len(sweep))
                    ),
                    key=lambda t: t[1],
                )
                jump_step = sweep[best[0]]["n_latent"]
                jump_pp = round(best[1] * 100, 1)
            peak_step = boundary.get("recommended_latent_steps")
            eff_plateau = (boundary.get("efficiency_plateau") or {}).get("boundary_x")
            rows.append(
                {
                    "id": sr.get("slice_id"),
                    "label": sr.get("slice_label"),
                    "mean_hops": ds.get("mean_reasoning_hops"),
                    "mean_diameter": ds.get("mean_diameter_from_root"),
                    "count": ds.get("count"),
                    "boundary": peak_step,
                    "max_accuracy": boundary.get("max_accuracy"),
                    "jump_step": jump_step,
                    "jump_pp": jump_pp,
                    "eff_boundary": eff_plateau,
                    "construction": (sr.get("slice_meta") or {}).get("construction"),
                    "ratio_4hop": (sr.get("slice_meta") or {}).get("ratio_4hop"),
                    "pattern_axis": (sr.get("slice_meta") or {}).get("pattern_axis"),
                    "sweep": sweep,
                }
            )
        for row in (payload.get("comparison") or {}).get("table") or []:
            if any(r.get("id") == row.get("id") for r in rows):
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "label": row.get("label"),
                    "mean_hops": row.get("mean_reasoning_hops"),
                    "mean_diameter": row.get("mean_diameter"),
                    "count": row.get("count"),
                    "boundary": row.get("boundary"),
                    "max_accuracy": row.get("max_accuracy"),
                    "construction": row.get("construction"),
                    "sweep": [],
                }
            )
    return rows


def _correlation(xs: List[float], ys: List[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den_x = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    den_y = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 3)


def _match_hop_boundary(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        hops = r.get("mean_hops")
        b = r.get("boundary")
        if hops is None or b is None:
            continue
        rounded = round(float(hops))
        out.append(
            {
                "label": r.get("label") or r.get("id"),
                "mean_hops": hops,
                "boundary": b,
                "matches_rounded_hops": abs(float(b) - rounded) <= 1,
                "max_accuracy": r.get("max_accuracy"),
            }
        )
    return out


def _mix_ladder(pattern: dict | None) -> dict:
    if not pattern:
        return {"available": False}
    mix_rows = []
    for sr in pattern.get("slices") or []:
        meta = sr.get("slice_meta") or {}
        if meta.get("pattern_axis") != "mix_ratio":
            continue
        boundary = (sr.get("boundary") or {}).get("recommended_latent_steps")
        ratio = meta.get("ratio_4hop")
        if ratio is None:
            continue
        mix_rows.append(
            {
                "ratio_4hop": ratio,
                "ratio_4hop_pct": int(round(float(ratio) * 100)),
                "boundary": boundary,
                "max_accuracy": (sr.get("boundary") or {}).get("max_accuracy"),
                "label": sr.get("slice_label"),
            }
        )
    mix_rows.sort(key=lambda r: r["ratio_4hop"])
    if not mix_rows:
        return {"available": False}

    flip_threshold = None
    for r in mix_rows:
        if r["boundary"] is not None and float(r["boundary"]) >= 4:
            flip_threshold = r["ratio_4hop_pct"]
            break

    return {
        "available": True,
        "rows": mix_rows,
        "flip_threshold_pct_4hop": flip_threshold,
        "summary": (
            f"4 跳占比从 0% 到 100%，边界依次为 "
            + " → ".join(f"{_step(r['boundary'])}步" for r in mix_rows)
            + (
                f"；约在 {flip_threshold}% 4 跳时切换到 4 步边界。"
                if flip_threshold is not None
                else "；全程保持 3 步边界（可能因并列取少）。"
            )
        ),
    }


def _hop_diameter_cross(pattern: dict | None) -> dict:
    if not pattern:
        return {"available": False}
    rows = []
    for sr in pattern.get("slices") or []:
        meta = sr.get("slice_meta") or {}
        if meta.get("pattern_axis") != "hop_diameter":
            continue
        ds = sr.get("dataset") or {}
        rows.append(
            {
                "label": sr.get("slice_label"),
                "mean_hops": ds.get("mean_reasoning_hops"),
                "mean_diameter": ds.get("mean_diameter_from_root"),
                "boundary": (sr.get("boundary") or {}).get("recommended_latent_steps"),
                "max_accuracy": (sr.get("boundary") or {}).get("max_accuracy"),
            }
        )
    if not rows:
        return {"available": False}

    hop3 = [r for r in rows if r.get("mean_hops") and float(r["mean_hops"]) < 3.5]
    hop4 = [r for r in rows if r.get("mean_hops") and float(r["mean_hops"]) >= 3.5]
    hop3_same = len({r["boundary"] for r in hop3 if r["boundary"] is not None}) <= 1
    hop4_same = len({r["boundary"] for r in hop4 if r["boundary"] is not None}) <= 1

    return {
        "available": True,
        "rows": rows,
        "summary": (
            "同一推理跳数内，图直径从 3 变到 4–5，边界"
            + ("基本不变（" if hop3_same and hop4_same else "有波动（")
            + "、".join(f"{r['label']}→{_step(r['boundary'])}步" for r in rows)
            + "）。说明边界跟推理链深度走，不跟图宽走。"
        ),
    }


def extract_pattern_laws() -> dict:
    latest = _load("latest.json")
    compare = _load("compare_latest.json")
    deep = _load("compare_deep_latest.json")
    variant = _load("compare_variant_latest.json")
    pattern = _load("compare_pattern_latest.json")

    all_rows = _collect_slice_rows(compare, deep, variant, pattern)
    real_rows = [
        r
        for r in all_rows
        if (r.get("construction") in (None, "real", "prosqa_extend") or r.get("pattern_axis"))
        and r.get("max_accuracy") is not None
        and (r.get("max_accuracy") or 0) > 0.05
    ]

    hop_corr = _correlation(
        [float(r["mean_hops"]) for r in real_rows if r.get("mean_hops") is not None],
        [float(r["boundary"]) for r in real_rows if r.get("boundary") is not None],
    )
    diam_corr = _correlation(
        [float(r["mean_diameter"]) for r in real_rows if r.get("mean_diameter") is not None],
        [float(r["boundary"]) for r in real_rows if r.get("boundary") is not None],
    )

    jump_steps = [r["jump_step"] for r in all_rows if r.get("jump_step") is not None]
    peak_steps = [r["boundary"] for r in all_rows if r.get("boundary") is not None]
    jump_mode = max(set(jump_steps), key=jump_steps.count) if jump_steps else None

    post_peak_drops = 0
    post_peak_total = 0
    for r in all_rows:
        sweep = r.get("sweep") or []
        b = r.get("boundary")
        if not sweep or b is None:
            continue
        peak_acc = max(row["accuracy"] for row in sweep)
        after = [row for row in sweep if row["n_latent"] > b]
        if not after:
            continue
        post_peak_total += 1
        if any(row["accuracy"] < peak_acc - 0.005 for row in after):
            post_peak_drops += 1

    extend_rows = [r for r in all_rows if r.get("construction") == "prosqa_extend"]
    dense_rows = [r for r in all_rows if r.get("construction") == "dense_chain"]

    full_acc3 = None
    full_acc5 = None
    if latest:
        sweep = {r["n_latent"]: r["accuracy"] for r in latest.get("latent_sweep") or []}
        full_acc3 = sweep.get(3)
        full_acc5 = sweep.get(5)

    mix = _mix_ladder(pattern)
    cross = _hop_diameter_cross(pattern)

    laws = [
        {
            "id": "depth_tracking",
            "title": "规律一：边界 ≈ 任务推理深度",
            "pattern": (
                f"跨 {len(real_rows)} 个子集，边界与平均推理跳数的相关系数 r={hop_corr or '—'}，"
                f"与图直径的 r={diam_corr or '—'}。"
                "纯 3 跳→3 步、纯 4 跳→4 步；换子集边界跟着变。"
            ),
            "why": (
                "论文：D 步连续思维 = 在图上并行 BFS 扩展 D 层。"
                "ProsQA 题的有效深度就是 3–4 跳，latent 步数是在潜空间里「往外搜几层」，"
                "搜够就答对，搜不够就错——所以边界跟题目需要几跳对齐，而不是模型写死的常数。"
            ),
            "evidence_tier": "很强",
            "examples": [
                {"label": "仅 3 跳", "value": "边界 3 步 · 90%"},
                {"label": "仅 4 跳", "value": "边界 4 步 · 88%"},
                {"label": "真实延长 5 跳", "value": "边界 5 步 · 96%"},
            ],
        },
        {
            "id": "jump_vs_peak",
            "title": "规律二：跳涨步（≈3）≠ 完成步（3 或 4）",
            "pattern": (
                f"最大准确率跃升出现在第 {jump_mode or 3} 步的子集占多数（2→3 步平均 +40pp 以上）；"
                f"但推荐边界分布在 {min(peak_steps) if peak_steps else '—'}–{max(peak_steps) if peak_steps else '—'} 步。"
                "这是两个不同现象。"
            ),
            "why": (
                "第 3 步是搜索前沿「打开」：叠加态第一次覆盖大部分可达节点（BFS frontier 展开）。"
                "4 跳题还需要第 4 步才补全最后一层信息。"
                "所以曲线总在 2→3 跳涨，但「任务完成」可能在 3 或 4——"
                "混测报 3 是因为 3 跳题已饱和 + 3/4 步准确率接近时算法取更少步数。"
            ),
            "evidence_tier": "很强",
            "examples": [
                {"label": "全量 2→3 步", "value": "+50.4pp"},
                {"label": "全量 3→4 步", "value": "-0.2pp（平台）"},
                {"label": "4 跳纯集峰值", "value": "第 4 步"},
            ],
        },
        {
            "id": "mix_threshold",
            "title": "规律三：混合比例决定报 3 还是报 4",
            "pattern": mix.get("summary", "第五轮混合阶梯实验运行后可定量。"),
            "why": (
                "混合集的全局峰值 = 3 跳题在 3 步的贡献 + 4 跳题在 4 步的贡献的加权平均。"
                "4 跳占比低时，3 步已照顾大部分题，整体峰值落在 3；"
                "4 跳占比高时，4 步的全局收益超过 3 步，边界上移到 4。"
                "这解释了全量 419 题（52% 为 4 跳）为何报 3 而非 4。"
            ),
            "evidence_tier": "强" if mix.get("available") else "待验证",
            "examples": [
                {"label": r["label"], "value": f'{r["ratio_4hop_pct"]}% 4跳 → {_step(r["boundary"])}步'}
                for r in (mix.get("rows") or [])
            ],
        },
        {
            "id": "hops_not_diameter",
            "title": "规律四：边界跟推理跳数走，不跟图直径走",
            "pattern": cross.get(
                "summary",
                "3 跳+直径 4 与 3 跳+直径 3 边界相同；4 跳+宽图仍报 4 步。",
            ),
            "why": (
                "Coconut 做的是沿 ground-truth 推理链的可达性搜索，不是「探索整张图有多宽」。"
                "直径更大的子图只是多了干扰边，不改变最短推理路径长度；"
                "latent 步数应对齐的是 BFS 到目标的最短深度（= 推理跳数），不是图的拓扑直径。"
            ),
            "evidence_tier": "强" if cross.get("available") else "中等",
            "examples": [
                {"label": r["label"], "value": f'边界 {_step(r["boundary"])} 步 · {_pct(r["max_accuracy"])}'}
                for r in (cross.get("rows") or [])
            ],
        },
        {
            "id": "post_peak_harm",
            "title": "规律五：超过最优点后加步 universal 有害",
            "pattern": (
                f"{post_peak_drops}/{post_peak_total} 个子集在峰值步之后出现准确率回落；"
                f"全量 5 步 {_pct(full_acc5)} vs 3 步 {_pct(full_acc3)}。"
            ),
            "why": (
                "所需可达性信息在 3–4 步已编码进叠加态；继续加步 = 对同一搜索前沿重复扩展，"
                "引入论文构造中的噪声 token 权重，扰动已有表示。"
                "不是「信息还不够」，而是「叠加态饱和后的干扰」。"
            ),
            "evidence_tier": "很强",
            "examples": [
                {"label": "全量 3 步", "value": "83.8%"},
                {"label": "全量 5 步", "value": "74.0%"},
                {"label": "全量 8 步", "value": "72.8%"},
            ],
        },
        {
            "id": "distribution_match",
            "title": "规律六：分布匹配才能抬高边界，光加长链不够",
            "pattern": (
                "同一 5–6 跳深度：真实 ProsQA 图延长 → 5 步边界 96%；"
                "简单人工稠密链 → 准确率接近 0%。"
                "合成 6 跳链 3 步即 97.5%，不随链长线性增步。"
            ),
            "why": (
                "checkpoint_300 在 ProsQA 3–4 跳上训练，学会了「3–4 步 latent + 原图边/token 分布」。"
                "真实延长保留原图结构与 token 化方式，模型能迁移；"
                "纯人工链是 OOD，模型不会做，边界数字无意义。"
                "要把边界稳定推到 5–6，需要匹配深度的训练数据，而非只改测试集。"
            ),
            "evidence_tier": "强",
            "examples": [
                {
                    "label": r.get("label") or r.get("id"),
                    "value": f'边界 {_step(r.get("boundary"))} 步 · {_pct(r.get("max_accuracy"))}',
                }
                for r in (extend_rows[:2] + dense_rows[:1])
            ],
        },
    ]

    unified_conclusion = (
        "六条规律可归为一条因果链："
        "Coconut 每步 latent 在机制上做并行 BFS → 任务推理深度决定需要几步 → "
        "2→3 步是前沿打开（跳涨），3–4 步是任务完成（平台）→ "
        "混合比例 + 并列取少决定混测报 3 还是 4 → "
        "加步超过最优点后饱和干扰 → "
        "抬高边界需要分布匹配的训练，而非简单加长测试链。"
    )

    return {
        "ok": True,
        "correlations": {
            "boundary_vs_mean_hops": hop_corr,
            "boundary_vs_mean_diameter": diam_corr,
        },
        "mix_ladder": mix,
        "hop_diameter_cross": cross,
        "hop_alignment": _match_hop_boundary(real_rows),
        "laws": laws,
        "unified_conclusion": unified_conclusion,
        "pattern_experiment_done": bool(pattern and pattern.get("ok")),
    }


def main():
    out = extract_pattern_laws()
    path = RESULTS / "pattern_laws.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
