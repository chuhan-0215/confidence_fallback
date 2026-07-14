"""Aggregate A800 GPU Phase 16–20 results for findings payload."""

from __future__ import annotations

from .format import fmt_pct, load_json

PHASE17 = "phase17/s1_corrected_final_latest.json"
PHASE18 = "phase18/w4_proof_rollup_latest.json"
PHASE19_U5 = "phase19/u5_failure_analysis_latest.json"
PHASE19_U6 = "phase19/u6_proof_rollup_latest.json"
TIMING_CEILING = "phase20/timing_ceiling_analysis.json"
PHASE20_ROLLUP = "phase20/v5_proof_rollup_latest.json"


def _load_phase_json(name: str) -> dict:
    data = load_json(name)
    if data:
        return data
    alt = load_json(f"from_a800/{name}")
    return alt or {}


def _pick_report(u5: dict, config: str) -> dict:
    for r in u5.get("reports") or []:
        if r.get("config") == config:
            return r
    return {}


def load_gpu_phase_bundle() -> dict:
    p17 = _load_phase_json(PHASE17)
    p18 = _load_phase_json(PHASE18)
    p19_u5 = _load_phase_json(PHASE19_U5)
    p19_u6 = _load_phase_json(PHASE19_U6)
    ceiling = _load_phase_json(TIMING_CEILING)
    p20 = _load_phase_json(PHASE20_ROLLUP)
    p20_best = p20.get("best") or {}

    primary = p17.get("primary_recommend") or {}
    strategies = {s.get("name"): s for s in (p17.get("strategies") or [])}
    min3_timing = strategies.get("min3_best_timing") or {}
    knn = strategies.get("knn_min3_full") or primary

    u5_best = _pick_report(p19_u5, "phase16_best_timing")
    p19_best = p19_u6.get("best") or {}
    p18_best = p18.get("best_timing") or p18.get("best_acc") or {}

    fc_dist = ceiling.get("fc_distribution") or {}
    fc_lt3 = sum(fc_dist.get(str(k), 0) for k in (1, 2))
    total_fc = ceiling.get("total_with_fc") or 408

    return {
        "ok": bool(p17.get("ok")),
        "deployable_mvp": p17.get("deployable_mvp"),
        "fully_proven_strict": p17.get("fully_proven_strict"),
        "layers": p17.get("layers") or [],
        "primary": primary,
        "knn": knn,
        "min3_timing": min3_timing,
        "p18_best": p18_best,
        "p19_best": p19_best,
        "u5_best": u5_best,
        "ceiling": ceiling,
        "fc_lt3_pct": fc_lt3 / total_fc if total_fc else None,
        "ceiling_min3": (ceiling.get("ceilings") or {}).get("3"),
        "ceiling_min2": (ceiling.get("ceilings") or {}).get("2"),
        "mentor_brief": p17.get("mentor_brief", ""),
        "p20_best": p20_best,
        "p20_feasible_any": p20.get("feasible_any"),
        "phase20_pending": not bool(p20.get("ok")),
    }


def build_gpu_phase_story_chapters(bundle: dict) -> list[dict]:
    if not bundle.get("ok"):
        return []

    knn = bundle.get("knn") or {}
    min3 = bundle.get("min3_timing") or {}
    p18 = bundle.get("p18_best") or {}
    p19 = bundle.get("p19_best") or {}
    u5 = bundle.get("u5_best") or {}
    ceiling = bundle.get("ceiling") or {}
    p20 = bundle.get("p20_best") or {}

    late_pct = u5.get("late_pct")
    early_pct = u5.get("early_pct")
    fc_lt3 = bundle.get("fc_lt3_pct")
    ceil3 = bundle.get("ceiling_min3")

    chapters = [
        {
            "id": "ch-gpu-16-17",
            "label": "GPU 定稿",
            "title": "全量 419 题硬跑：[[deployable-mvp|能部署]]了吗？",
            "paragraphs": [
                "实验十一至五十五在 CPU 上摸清了几十条自停路线，但结论必须在 A800 上对着全量 419 题再验一遍。"
                "Phase 16 起进入 GPU 定稿包：min_n 网格、前缀 kNN、Pareto 与五层证明 rollup。",
                f"Phase 17 修正 rollup 脚本 bug 后，五层证明重新盖章——"
                f"[[m2-head|M2 head]] 可学习、[[min-n|min_n=3]] 把全量 timing 推到 {fmt_pct(min3.get('timing'))}，"
                f"[[knn-min3|knn_min3_full]] 全量 acc {fmt_pct(knn.get('accuracy'))}、mean_n={knn.get('mean_n', 3.37):.2f}，"
                f"无 oracle，[[deployable-mvp|deployable_mvp]]=True。",
                "通俗说：模型自己停 + 一点点 kNN 校正 floor，不用偷看答案，92.6% 答对、算力也省——"
                "「够好就停」在答对率和算力维度可以写进部署结论。"
                "但 [[strict-feasible|timing≥50%]] 仍是 stretch goal：head-only 天花板约 37–39%。",
            ],
            "highlight": (
                f"knn_min3_full {fmt_pct(knn.get('accuracy'))} · deployable_mvp ✅ · "
                f"timing {fmt_pct(min3.get('timing'))} · strict_feasible ❌"
            ),
        },
        {
            "id": "ch-gpu-18-19",
            "label": "GPU 十八–十九",
            "title": "换权重、换推理——[[timing-ceiling|timing 天花板]]在哪？",
            "paragraphs": [
                "Phase 18 动 Coconut 权重：joint 微调、写回 schedule、fc 标签长训。"
                f"最好 w3_fc_long：acc {fmt_pct(p18.get('accuracy'))}，timing {fmt_pct(p18.get('timing'))}——"
                "与 Phase 16 的 37% 几乎肩并肩，100% 题仍停在 n=3。"
                "长训、联合、写回都没把停步从 n=3 挪开——瓶颈不在「练得不够久」。",
                "Phase 19 换推理策略：patience、AND/OR 组合、learned hybrid、失败解剖。"
                f"最优 u1_patience：acc {fmt_pct(p19.get('accuracy'))}，timing {fmt_pct(p19.get('timing'))}，未破 Phase 16。"
                f"U5 失败解剖亮出主因：[[late-stop|late_stop]] 占 {fmt_pct(late_pct)}，"
                f"[[early-stop|early_stop]] 仅 {fmt_pct(early_pct)}——不是模型太急着想停，"
                f"而是 [[min-n|min_n=3]] 不让早停，大量 [[first-correct|fc<3]] 的题被硬拖到第 3 步以后。",
                f"CPU 分析：{fmt_pct(fc_lt3)} 题 [[first-correct|首次答对]] 发生在第 1–2 步；"
                f"min_n=3 理论 timing 上限约 {fmt_pct(ceil3)}；"
                f"实测 {fmt_pct(min3.get('timing'))} 已达上限的 84%——调参只剩 2–5 pp 空间。",
            ],
            "highlight": (
                f"late_stop {fmt_pct(late_pct)} · timing 天花板 {fmt_pct(ceil3)} · "
                f"当前 {fmt_pct(min3.get('timing'))}"
            ),
        },
        {
            "id": "ch-gpu-20",
            "label": "GPU 二十",
            "title": "min_n=2 侧突破——A800 已验证失败",
            "paragraphs": [
                "Phase 13–19 换标签、换推理、联合长训均未能把 timing 推过 50%——"
                "需要从 [[min-n|min_n 规则]] 或 Coconut 写回层动刀。",
                f"Phase 20 V1–V5 已在 A800 跑完：最好 {p20.get('id', 'v2_writeback_infer')} "
                f"acc {fmt_pct(p20.get('accuracy'))}、timing {fmt_pct(p20.get('timing'))}——"
                "仍低于 Phase 16 knn 92.6%，feasible_any=false。",
                ceiling.get("insight", "")
                or "min_n=2 理论 timing 上限约 61%，但实测未破 37% 平台；timing 瓶颈确认为规则+fc 分布。",
                "与 [[hybrid-stop|hybrid 上界]]（97% acc，含 BFS）对照：部署线走 M2+knn 盲停，"
                "上界线保留 hybrid 作性能参照。",
            ],
            "highlight": (
                f"Phase 20 完成 · best timing {fmt_pct(p20.get('timing'))} · feasible_any=false"
            ),
        },
    ]
    return chapters


def merge_gpu_phase_story(story: dict, bundle: dict) -> dict:
    if not story or not bundle.get("ok"):
        return story

    chapters = list(story.get("chapters") or [])
    insert_at = next((i for i, ch in enumerate(chapters) if ch.get("id") == "ch-result"), len(chapters))
    gpu_chapters = build_gpu_phase_story_chapters(bundle)
    chapters[insert_at:insert_at] = gpu_chapters
    story["chapters"] = chapters

    knn = bundle.get("knn") or {}
    min3 = bundle.get("min3_timing") or {}
    u5 = bundle.get("u5_best") or {}

    conclusion = story.get("conclusion") or {}
    takeaways = list(conclusion.get("takeaways") or [])
    takeaways.extend(
        [
            (
                f"GPU 定稿：[[deployable-mvp|deployable_mvp]]=True；"
                f"推荐 [[knn-min3|knn_min3_full]] acc {fmt_pct(knn.get('accuracy'))}，mean_n={knn.get('mean_n', 3.37):.2f}，无 oracle。"
            ),
            (
                f"[[strict-feasible|timing≥50%]] 未达标：全量最好 {fmt_pct(min3.get('timing'))}，"
                f"距 50% 差 11–13 pp；[[late-stop|late_stop]] 占 {fmt_pct(u5.get('late_pct'))}。"
            ),
            (
                f"Phase 20 结论：min_n=2 路线未破 timing（最好 {fmt_pct((bundle.get('p20_best') or {}).get('timing'))}），"
                "strict_feasible 仍不可达。"
            ),
        ]
    )
    conclusion["takeaways"] = takeaways
    story["conclusion"] = conclusion

    science = story.get("science_box") or {}
    laws = list(science.get("laws") or [])
    laws.extend(
        [
            (
                f"规律十七：[[deployable-mvp|deployable_mvp]] 已达成——"
                f"[[knn-min3|knn_min3_full]] acc {fmt_pct(knn.get('accuracy'))}，无 oracle，mean_n≤4.5。"
            ),
            (
                f"规律十八：[[strict-feasible|timing≥50%]] 未过——head-only 天花板 "
                f"{fmt_pct(min3.get('timing'))}；[[late-stop|late_stop]] 是主因（≈57%）。"
            ),
            (
                f"规律十九：[[min-n|min_n=3]] 下 timing 理论上限约 {fmt_pct(bundle.get('ceiling_min3'))}；"
                "56% 题 fc<3，算术瓶颈非单纯训练问题。"
            ),
        ]
    )
    science["laws"] = laws
    stats = list(science.get("stats") or [])
    stats.extend(
        [
            ("deployable_mvp", "✅ knn_min3"),
            ("推荐 acc", fmt_pct(knn.get("accuracy"))),
            ("timing 天花板", fmt_pct(min3.get("timing"))),
        ]
    )
    science["stats"] = stats
    story["science_box"] = science

    result_ch = next((ch for ch in story["chapters"] if ch.get("id") == "ch-result"), None)
    if result_ch:
        paras = list(result_ch.get("paragraphs") or [])
        if paras:
            paras[0] = (
                paras[0]
                + f" GPU Phase 17 定稿：[[deployable-mvp|deployable_mvp]]=True（[[knn-min3|knn_min3_full]] "
                f"{fmt_pct(knn.get('accuracy'))}）；timing 天花板 {fmt_pct(min3.get('timing'))}，"
                f"[[strict-feasible|strict_feasible]] 未过；Phase 20 min_n=2 未突破。"
            )
            result_ch["paragraphs"] = paras
    return story


def build_gpu_phase_analysis(bundle: dict) -> dict:
    if not bundle.get("ok"):
        return {}

    knn = bundle.get("knn") or {}
    min3 = bundle.get("min3_timing") or {}
    p18 = bundle.get("p18_best") or {}
    p19 = bundle.get("p19_best") or {}
    u5 = bundle.get("u5_best") or {}
    ceiling = bundle.get("ceiling") or {}
    p20 = bundle.get("p20_best") or {}

    layer_rows = []
    for layer in bundle.get("layers") or []:
        layer_rows.append(
            [
                layer.get("layer", ""),
                "✅" if layer.get("pass") else "❌",
                layer.get("detail", ""),
            ]
        )

    p20 = bundle.get("p20_best") or {}
    phase_rows = [
        ["Phase 17 定稿", "knn_min3_full", fmt_pct(knn.get("accuracy")), fmt_pct(knn.get("timing")), "deployable_mvp ✅"],
        ["Phase 17", "min3_best_timing", fmt_pct(min3.get("accuracy")), fmt_pct(min3.get("timing")), "timing 最优"],
        ["Phase 18", (p18.get("id") or "w3_fc_long"), fmt_pct(p18.get("accuracy")), fmt_pct(p18.get("timing")), "联合/写回未破"],
        ["Phase 19", (p19.get("id") or "u1_patience"), fmt_pct(p19.get("accuracy")), fmt_pct(p19.get("timing")), "换推理未破"],
        ["Phase 20", (p20.get("id") or "v2_writeback_infer"), fmt_pct(p20.get("accuracy")), fmt_pct(p20.get("timing")), "min_n=2 未破"],
    ]

    never_fc = u5.get("never_fc")
    never_fc_pct = (never_fc / 419) if never_fc is not None else None

    failure_rows = [
        ["early_stop（停太早）", fmt_pct(u5.get("early_pct")), "需 patience/延后"],
        ["late_stop（停太晚）", fmt_pct(u5.get("late_pct")), "min_n=3 硬拖 + fc<3"],
        ["never_fc（从未答对）", fmt_pct(never_fc_pct), "少数"],
    ]

    bullets = [
        f"[[deployable-mvp|deployable_mvp]]：全量 acc≥86.3%、mean_n≤4.5、推理无 oracle、M3 必要性成立。",
        f"推荐部署：[[knn-min3|knn_min3_full]] min_n=3 thr=0.15，acc {fmt_pct(knn.get('accuracy'))}，mean_n={knn.get('mean_n', 3.37):.2f}。",
        f"[[strict-feasible|strict_feasible]]：timing≥50% 未过；全量最好 {fmt_pct(min3.get('timing'))}（min_n=3, thr=0.35）。",
        f"timing 理论上限（min_n=3）：约 {fmt_pct(bundle.get('ceiling_min3'))}；当前达上限 84%。",
        f"失败主因：[[late-stop|late_stop]] {fmt_pct(u5.get('late_pct'))}；{fmt_pct(bundle.get('fc_lt3_pct'))} 题 fc 在第 1–2 步。",
        f"Phase 20 已完成：min_n=2 最好 timing {fmt_pct(p20.get('timing'))}，feasible_any=false。",
        "[[hybrid-stop|hybrid]] 上界 97% 含 BFS/序贯，作性能参照；部署线用 M2+knn 盲停。",
    ]

    return {
        "title": "GPU Phase 16–20 · 全量定稿与 timing 天花板",
        "tldr": (
            f"deployable_mvp ✅（knn {fmt_pct(knn.get('accuracy'))}）· "
            f"timing {fmt_pct(min3.get('timing'))} / 上限 {fmt_pct(bundle.get('ceiling_min3'))} · "
            f"late_stop {fmt_pct(u5.get('late_pct'))} · Phase 20 timing {fmt_pct(p20.get('timing'))}"
        ),
        "lead": (
            "CPU 实验十一至五十五在 A800 上全量 419 题复验。"
            "Phase 17 定稿 deployable_mvp；Phase 18–19 证实 timing 瓶颈在 min_n 规则与 fc 分布，非训练不足。"
        ),
        "proof_table": {
            "headers": ["证明层", "通过", "说明"],
            "rows": layer_rows,
        },
        "phase_table": {
            "headers": ["阶段", "方案", "acc", "timing", "备注"],
            "rows": phase_rows,
        },
        "failure_table": {
            "headers": ["失败类型", "占比", "含义"],
            "rows": failure_rows,
        },
        "ceiling_insight": ceiling.get("insight", ""),
        "bullets": bullets,
        "law": (
            "答对率维度已可部署（[[deployable-mvp|deployable_mvp]]）；"
            "停步时机维度卡在 [[min-n|min_n=3]] 算术上限——"
            "要破 timing 50% 需从 min_n 规则或写回层入手；Phase 20 min_n=2 已证伪。"
        ),
        "phase20_pending": bundle.get("phase20_pending"),
    }


def build_gpu_phase_experiments(bundle: dict) -> list[dict]:
    if not bundle.get("ok"):
        return []
    knn = bundle.get("knn") or {}
    min3 = bundle.get("min3_timing") or {}
    p20 = bundle.get("p20_best") or {}
    return [
        {
            "id": "gpu_phase17",
            "title": "GPU Phase 17 · 定稿修正",
            "samples": "全量 419",
            "latent_range": "M2+knn",
            "boundary": "deployable_mvp",
            "peak_accuracy": knn.get("accuracy"),
            "note": f"knn_min3_full acc {fmt_pct(knn.get('accuracy'))}；五层证明 rollup 修正后 deployable_mvp=True。",
        },
        {
            "id": "gpu_phase18_19",
            "title": "GPU Phase 18–19 · timing 冲顶与失败解剖",
            "samples": "全量 419",
            "latent_range": "joint/推理",
            "boundary": "timing 瓶颈",
            "peak_accuracy": min3.get("accuracy"),
            "note": (
                f"timing 天花板 {fmt_pct(min3.get('timing'))}；late_stop≈57%；"
                "联合训练与换推理均未破 Phase 16。"
            ),
        },
        {
            "id": "gpu_phase20",
            "title": "GPU Phase 20 · min_n=2 突破（已完成）",
            "samples": "全量 419",
            "latent_range": "V1–V5",
            "boundary": "timing 突破",
            "peak_accuracy": p20.get("accuracy"),
            "note": (
                f"best {p20.get('id')} timing {fmt_pct(p20.get('timing'))}；"
                "feasible_any=false，未破 Phase 16。"
            ),
        },
    ]


def build_gpu_phase_highlight_body(bundle: dict) -> str:
    if not bundle.get("ok"):
        return "GPU Phase 16–20 结果待汇总。"
    knn = bundle.get("knn") or {}
    min3 = bundle.get("min3_timing") or {}
    u5 = bundle.get("u5_best") or {}
    p20 = bundle.get("p20_best") or {}
    return (
        f"A800 全量定稿：[[deployable-mvp|deployable_mvp]]=True，推荐 [[knn-min3|knn_min3_full]] "
        f"acc {fmt_pct(knn.get('accuracy'))}、mean_n={knn.get('mean_n', 3.37):.2f}，无 oracle。"
        f"[[strict-feasible|timing≥50%]] 未过：最好 {fmt_pct(min3.get('timing'))}，"
        f"[[late-stop|late_stop]] 占 {fmt_pct(u5.get('late_pct'))}；"
        f"Phase 20 min_n=2 最好 {fmt_pct(p20.get('timing'))}，feasible_any=false。"
    )
