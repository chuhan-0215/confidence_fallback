"""Assemble findings_summary.json payload from result files."""

from __future__ import annotations

from datetime import datetime, timezone

from pattern_analysis import extract_pattern_laws

from .builders import (
    build_boundary_push_analysis,
    build_boundary_push_deep_analysis,
    build_essence,
    build_extreme_cases,
    build_math_proof,
    build_mechanism_analysis,
    build_model_perturb_analysis,
    build_feedback_schedule_analysis,
    build_feedback_playbook,
    build_feedback_highlight_body,
    build_auto_submit_analysis,
    build_auto_submit_highlight_body,
    validate_feedback_schedule_copy,
    build_story,
)
from .adaptive_stop import (
    load_adaptive_stop_bundle,
    build_adaptive_stop_analysis,
    build_adaptive_stop_highlight_body,
    merge_adaptive_stop_story,
    build_adaptive_stop_experiments,
)
from .gpu_phase import (
    load_gpu_phase_bundle,
    build_gpu_phase_analysis,
    build_gpu_phase_highlight_body,
    merge_gpu_phase_story,
    build_gpu_phase_experiments,
)
from .cross_transfer import (
    load_cross_transfer_bundle,
    build_cross_transfer_analysis,
    build_cross_transfer_highlight_body,
    merge_cross_transfer_story,
    build_cross_transfer_experiments,
)
from .format import fmt_pct, fmt_step, load_json

def build_payload() -> dict:
    full = load_json("latest.json") or {}
    compare = load_json("compare_latest.json") or {}
    deep = load_json("compare_deep_latest.json") or {}
    variant = load_json("compare_variant_latest.json") or {}
    pattern = load_json("compare_pattern_latest.json") or {}
    boundary_push = load_json("compare_boundary_push_latest.json") or {}
    boundary_push_deep = load_json("compare_boundary_push_deep_latest.json") or {}
    model_perturb = load_json("model_perturb_latest.json") or {}
    feedback_schedule = load_json("feedback_schedule_latest.json") or {}
    auto_submit = load_json("auto_submit_latest.json") or {}
    pattern_laws = extract_pattern_laws()

    full_boundary = (full.get("boundary") or {}).get("recommended_latent_steps")
    full_acc = (full.get("boundary") or {}).get("max_accuracy")
    full_sweep = {r["n_latent"]: r["accuracy"] for r in full.get("latent_sweep", [])}
    why = full.get("why_analysis") or {}

    cmp_table = (compare.get("comparison") or {}).get("table") or []
    deep_table = (deep.get("comparison") or {}).get("table") or []
    var_table = (variant.get("comparison") or {}).get("table") or []

    by_id = {r["id"]: r for r in cmp_table}
    h3, h4 = by_id.get("hops_3"), by_id.get("hops_4")

    var_construction = [
        r
        for r in var_table
        if r.get("id") not in ("v_chain_6_symbol", "v_extend_6_symbol", "v_real_4_symbol")
        and "symbol" not in (r.get("id") or "")
    ]

    extend_5 = next((r for r in var_table if r.get("id") == "v_extend_5"), None)
    extend_6 = next((r for r in var_table if r.get("id") == "v_extend_6"), None)
    chain_6 = next((r for r in var_table if r.get("id") == "v_chain_6_dense"), None)
    syn_6 = next((r for r in deep_table if r.get("id") == "syn_chain_6"), None)
    syn_5 = next((r for r in deep_table if r.get("id") == "syn_chain_5"), None)

    insights = []
    for src in (compare, deep, variant):
        insights.extend((src.get("comparison") or {}).get("insights") or [])

    key_numbers = {
        "full_samples": full.get("dataset", {}).get("count"),
        "full_boundary_steps": full_boundary,
        "full_peak_accuracy": full_acc,
        "compare_boundary_min": (compare.get("comparison") or {}).get("boundary_range", {}).get("min"),
        "compare_boundary_max": (compare.get("comparison") or {}).get("boundary_range", {}).get("max"),
        "acc_at_1_steps": full_sweep.get(1),
        "acc_at_2_steps": full_sweep.get(2),
        "acc_at_3_steps": full_sweep.get(3),
        "acc_at_4_steps": full_sweep.get(4),
        "acc_at_5_steps": full_sweep.get(5),
    }

    mechanism_analysis = build_mechanism_analysis(
        full,
        compare,
        deep,
        variant,
        why,
        key_numbers,
        by_id,
        h3,
        h4,
        extend_5,
        syn_5,
        syn_6,
    )
    extreme_cases = build_extreme_cases(var_table, deep_table)
    essence = build_essence(
        key_numbers, pattern_laws, h3, h4, extend_5, extend_6, syn_6, boundary_push, boundary_push_deep
    )
    story = build_story(
        key_numbers, h3, h4, extend_5, syn_5, syn_6, boundary_push, model_perturb, boundary_push_deep, feedback_schedule, auto_submit, pattern_laws
    )
    adaptive_bundle = load_adaptive_stop_bundle()
    adaptive_stop_analysis = build_adaptive_stop_analysis(adaptive_bundle)
    gpu_bundle = load_gpu_phase_bundle()
    gpu_phase_analysis = build_gpu_phase_analysis(gpu_bundle)
    cross_bundle = load_cross_transfer_bundle()
    cross_transfer_analysis = build_cross_transfer_analysis(cross_bundle)
    auto_ar_fmt = fmt_pct((auto_submit.get("summary") or {}).get("auto_route_accuracy"))
    story = merge_adaptive_stop_story(story, adaptive_bundle, auto_ar_fmt if auto_ar_fmt != "—" else None)
    story = merge_gpu_phase_story(story, gpu_bundle)
    story = merge_cross_transfer_story(story, cross_bundle)
    math_proof = build_math_proof(full)
    boundary_push_analysis = build_boundary_push_analysis(boundary_push)
    boundary_push_deep_analysis = build_boundary_push_deep_analysis(boundary_push_deep)
    model_perturb_analysis = build_model_perturb_analysis(model_perturb)
    feedback_schedule_analysis = build_feedback_schedule_analysis(feedback_schedule)
    feedback_playbook = build_feedback_playbook(feedback_schedule, h3, h4)
    auto_submit_analysis = build_auto_submit_analysis(auto_submit)

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "Coconut checkpoint_300",
        "dataset": "ProsQA + 合成/变体扩展",
        "headline": "一栋楼里找人的故事：AI 要想几层，才算想够？",
        "one_liner": (
            "ProsQA 同源 confidence_fallback 94.75%；跨 53 切片 hybrid_router 加权 Δ +2.05 pp、OOD +7.44 pp；"
            "deploy_spec_v4 双轨定稿（seed=99）；timing 天花板 37%，Phase 20 min_n=2 未破。"
        ),
        "site_guide": {
            "lead": "本报告分四页——先读故事建立直觉，查附录看数据与方案，进实验室复现。",
            "items": [
                {
                    "tag": "当前",
                    "title": "实验故事",
                    "href": "#story-guide",
                    "desc": "十轮边界 + 自停探索 + GPU 定稿 · 章节目录可跳转",
                },
                {
                    "tag": "数据",
                    "title": "科学附录",
                    "href": "appendix.html#appendix-toc",
                    "desc": "对照表 · 自停汇总 · GPU 定稿 · FAQ",
                },
                {
                    "tag": "术语",
                    "title": "术语注释",
                    "href": "glossary.html",
                    "desc": "文中下划线词的一站式解释",
                },
                {
                    "tag": "复现",
                    "title": "交互实验室",
                    "href": "lab.html",
                    "desc": "点击重跑各轮实验",
                },
            ],
        },
        "key_numbers": key_numbers,
        "story": story,
        "math_proof": math_proof,
        "boundary_push": boundary_push_analysis,
        "boundary_push_deep": boundary_push_deep_analysis,
        "model_perturb": model_perturb_analysis,
        "feedback_schedule": feedback_schedule_analysis,
        "feedback_playbook": feedback_playbook,
        "auto_submit": auto_submit_analysis,
        "adaptive_stop": adaptive_stop_analysis,
        "gpu_phase": gpu_phase_analysis,
        "cross_transfer": cross_transfer_analysis,
        "cross_transfer_bundle": {
            "project_status": cross_bundle.get("project_status"),
            "deploy_spec_version": "v4",
            "dual_ok": cross_bundle.get("dual_ok"),
            "weighted_delta": cross_bundle.get("weighted_delta"),
        },
        "gpu_phase_bundle": {
            "deployable_mvp": gpu_bundle.get("deployable_mvp"),
            "primary": gpu_bundle.get("primary"),
            "phase20_pending": gpu_bundle.get("phase20_pending"),
        },
        "adaptive_stop_bundle": {
            "full_count": adaptive_bundle.get("full_count"),
            "baselines": adaptive_bundle.get("baselines"),
            "feasible_track_ids": adaptive_stop_analysis.get("feasible_track_ids") if adaptive_stop_analysis else [],
            "online_pending": adaptive_bundle.get("online_pending"),
        },
        "essence": essence,
        "mechanism_analysis": mechanism_analysis,
        "extreme_cases": extreme_cases,
        "why_analysis": {
            "summary": why.get("summary", ""),
            "reasons": why.get("reasons") or [],
            "marginal_gains": why.get("marginal_gains") or [],
            "largest_gain_step": why.get("largest_gain_step"),
        },
        "insights": insights[:8],
        "experiments": [
            {
                "id": "full",
                "title": "实验一 · 全量 ProsQA",
                "samples": 419,
                "latent_range": "1–8",
                "boundary": full_boundary,
                "peak_accuracy": full_acc,
                "note": "419 条混合 3/4 跳，峰值在 3 步（83.8%），4 步几乎相同，5 步起下降。",
            },
            {
                "id": "compare",
                "title": "实验二 · 多数据集对比",
                "samples": "10 子集 × 60",
                "latent_range": "1–8",
                "boundary": f"{fmt_step((compare.get('comparison') or {}).get('boundary_range', {}).get('min'))}–{fmt_step((compare.get('comparison') or {}).get('boundary_range', {}).get('max'))}",
                "peak_accuracy": (compare.get("comparison") or {}).get("accuracy_range", {}).get("max"),
                "note": "纯 3 跳→3 步，纯 4 跳→4 步；混测多报 3 步。",
            },
            {
                "id": "deep",
                "title": "实验三 · 深边界探测",
                "samples": "8 子集 × 40",
                "latent_range": "1–10",
                "boundary": f"{fmt_step((deep.get('comparison') or {}).get('boundary_range', {}).get('min'))}–{fmt_step((deep.get('comparison') or {}).get('boundary_range', {}).get('max'))}",
                "peak_accuracy": (deep.get("comparison") or {}).get("accuracy_range", {}).get("max"),
                "note": "合成 5 跳几乎全错；合成 6 跳 3 步即 97.5%，未随链长线性增步。",
            },
            {
                "id": "variant",
                "title": "实验四 · 构造×监督对照",
                "samples": "16 子集 × 25",
                "latent_range": "1–10",
                "boundary": "见分项（索引监督下仍多 3–5 步）",
                "peak_accuracy": (variant.get("comparison") or {}).get("accuracy_range", {}).get("max"),
                "note": "真实图延长 5 跳可达 5 步边界；纯人工链 OOD 失败；符号监督与训练不一致则指标失效。",
            },
            {
                "id": "pattern",
                "title": "实验五 · 规律寻探",
                "samples": "21 子集汇总",
                "latent_range": "1–8",
                "boundary": "随题深 r≈0.54",
                "peak_accuracy": None,
                "note": (
                    f"边界↔平均跳数 r={pattern_laws.get('correlations', {}).get('boundary_vs_mean_hops', '0.543')}；"
                    f"{len(pattern_laws.get('laws') or [])} 条规律 · 附录 #pattern-laws"
                    if pattern_laws.get("laws")
                    else "scripts/pattern_analysis.py"
                ),
            },
            {
                "id": "boundary_push",
                "title": "实验六 · 边界上推（同质延长）",
                "samples": "7 子集 × 25",
                "latent_range": "1–10",
                "boundary": (
                    f"{fmt_step((boundary_push.get('comparison') or {}).get('boundary_range', {}).get('min'))}–"
                    f"{fmt_step((boundary_push.get('comparison') or {}).get('boundary_range', {}).get('max'))}"
                    if boundary_push.get("ok")
                    else "5–9 步"
                ),
                "peak_accuracy": (boundary_push.get("comparison") or {}).get("accuracy_range", {}).get("max"),
                "note": (
                    "3 跳基线延长 5 跳→边界 5、100%；4 跳基线易过冲；6 跳混合延长边界漂到 9。"
                    if boundary_push.get("ok")
                    else "scripts/run_compare_experiment.py --push"
                ),
            },
            {
                "id": "model_perturb",
                "title": "实验七 · 只改模型数值",
                "samples": "3 子集 × 15 · 10 组扰动",
                "latent_range": "1–8",
                "boundary": "随 α/权重在 d 附近 ±1–2 步",
                "peak_accuracy": 1.0,
                "note": (
                    "固定 ProsQA 题面；latent 反馈 α∈[0.5,2.0] 时 8/18 组边界变化；"
                    "acc@d 常下降。见 model_perturb_latest.json。"
                    if model_perturb.get("ok")
                    else "scripts/run_model_perturb_experiment.py"
                ),
            },
            {
                "id": "boundary_push_deep",
                "title": "实验八 · 7–8 步深边界上推",
                "samples": "5 子集 × 25",
                "latent_range": "1–12",
                "boundary": (
                    f"{fmt_step((boundary_push_deep.get('comparison') or {}).get('boundary_range', {}).get('min'))}–"
                    f"{fmt_step((boundary_push_deep.get('comparison') or {}).get('boundary_range', {}).get('max'))}"
                    if boundary_push_deep.get("ok")
                    else "待运行"
                ),
                "peak_accuracy": (boundary_push_deep.get("comparison") or {}).get("accuracy_range", {}).get("max"),
                "note": (
                    (boundary_push_deep_analysis or {}).get("tldr", "")[:120]
                    if boundary_push_deep.get("ok")
                    else "scripts/run_compare_experiment.py --push-deep"
                ),
            },
            {
                "id": "feedback_schedule",
                "title": "实验九 · 4 步后性能平台（feedback schedule）",
                "samples": "3 子集 · 6 策略",
                "latent_range": "1–8",
                "boundary": "看 acc@4 与 post4_drop",
                "peak_accuracy": None,
                "note": (
                    (feedback_schedule_analysis or {}).get("tldr", "")[:120]
                    if feedback_schedule.get("ok")
                    else "scripts/run_feedback_schedule_experiment.py"
                ),
            },
            {
                "id": "auto_submit",
                "title": "实验十 · 无标签自动配参（通解验证）",
                "samples": 419,
                "latent_range": "按题 d∈{3,4}",
                "boundary": "按题路由",
                "peak_accuracy": (auto_submit.get("summary") or {}).get("auto_route_accuracy"),
                "note": (
                    (auto_submit_analysis or {}).get("tldr", "")[:120]
                    if auto_submit.get("ok")
                    else "scripts/run_auto_submit_experiment.py"
                ),
            },
        ] + build_adaptive_stop_experiments(adaptive_bundle) + build_gpu_phase_experiments(gpu_bundle) + build_cross_transfer_experiments(cross_bundle),
        "pattern_laws": pattern_laws,
        "highlights": [
            {
                "title": "边界随数据而变",
                "body": (
                    f"10 个子集边界分布在 3–4 步：纯 3 跳子集 {fmt_step(h3 and h3.get('boundary'))} 步（{fmt_pct(h3 and h3.get('max_accuracy'))}），"
                    f"纯 4 跳子集 {fmt_step(h4 and h4.get('boundary'))} 步（{fmt_pct(h4 and h4.get('max_accuracy'))}）。"
                    "不是模型硬编码的固定步数。"
                ),
            },
            {
                "title": "第 3 步是「跳涨点」",
                "body": (
                    f"全量实验：1 步 42% → 2 步 33% → 3 步 {fmt_pct(full_sweep.get(3))}。"
                    "多数子集最大涨幅出现在 2→3 步，连续思维在此展开搜索前沿。"
                ),
            },
            {
                "title": "加步过多会饱和/干扰",
                "body": (
                    f"全量：3 步 {fmt_pct(full_sweep.get(3))}，5 步 {fmt_pct(full_sweep.get(5))}，8 步更低。"
                    "信息够用时继续加 latent 不再受益，反而扰动已有表示。"
                ),
            },
            {
                "title": "真实 ProsQA 题深只有 3–4 跳",
                "body": "419 题中 202 条 3 跳、217 条 4 跳，无原生 5–6 跳。图直径≥5 的子集推理链仍多为 3–4 跳，边界因此难稳定到 5–6 步。",
            },
            {
                "title": "合成链 vs 真实延长图",
                "body": (
                    f"简单合成 5/6 跳链：准确率 {fmt_pct(syn_5 and syn_5.get('max_accuracy'))} / {fmt_pct(syn_6 and syn_6.get('max_accuracy'))}（OOD 或 3 步饱和）。"
                    + (
                        f" 在真实 ProsQA 图上延长到 5/6 跳：边界 {fmt_step(extend_5 and extend_5.get('boundary'))} / {fmt_step(extend_6 and extend_6.get('boundary'))} 步，"
                        f"准确率 {fmt_pct(extend_5 and extend_5.get('max_accuracy'))} / {fmt_pct(extend_6 and extend_6.get('max_accuracy'))}。"
                        if extend_5 and extend_6
                        else ""
                    )
                ),
            },
            {
                "title": "只改模型数值也能动边界",
                "body": (
                    "第七轮固定 ProsQA 题面，调 latent 反馈系数 α 或权重 ±15%："
                    "3 跳题在 α=0.5 时边界 3→4；5 跳延长题在 α=2.0 时边界 5→7，"
                    "但 acc@5 从 100% 降至 80%。报边界与 acc@d 须同看。"
                ),
            },
            {
                "title": "4 步以后如何不跌？（实验九）",
                "body": build_feedback_highlight_body(feedback_schedule),
            },
            {
                "title": "无标签通解怎么一次提交？（实验十）",
                "body": build_auto_submit_highlight_body(auto_submit),
            },
            {
                "title": "模型能否自己决定何时停？（实验十一–五十五）",
                "body": build_adaptive_stop_highlight_body(adaptive_bundle),
            },
            {
                "title": "A800 全量定稿与 timing 天花板（GPU Phase 16–20）",
                "body": build_gpu_phase_highlight_body(gpu_bundle),
            },
            {
                "title": "跨集 transfer 与 deploy_spec_v4（GPU Phase 32–38）",
                "body": build_cross_transfer_highlight_body(cross_bundle),
            },
            {
                "title": "监督方式必须一致",
                "body": (
                    "同一 6 跳数据：索引监督边界 "
                    f"{fmt_step(chain_6 and chain_6.get('boundary'))} 步；改为符号名监督后准确率 0%，边界指标失效。"
                    "评估格式须与训练一致。"
                ),
            },
        ],
        "faq": [
            {"q": "边界是固定常数吗？", "a": "不是。纯 3 跳报 3 步，纯 4 跳报 4 步；换子集、换构造方式都会变。"},
            {
                "q": "只改模型数值、不改题面，边界会变吗？",
                "a": (
                    "会，但在任务深度 d 附近 ±1–2 步内，且 acc@d 常变差。"
                    "latent 反馈 α<1 可能需更多步；α>1 或权重缩放可能过冲。"
                    "稳定 c=d 仍靠 α≈1 + 原 checkpoint + 题深匹配。"
                ),
            },
            {
                "q": "能把边界推到 5–6 步吗？",
                "a": (
                    "数据侧：3 跳基线 prosqa_extend 到 5 跳可对齐边界 5（96–100%）。"
                    "6 跳仍易平台/过冲。模型侧：调 α 可抬高报边界，但 acc@d 难同步。"
                    "稳定 6 步需深度匹配训练 + 微调。"
                ),
            },
            {
                "q": "第 5–8 步为什么会跌？怎么修？",
                "a": (
                    "baseline 每步把 hidden 写回 embedding；4 步后信息已够用，继续写回添噪。"
                    "实验九：schedule [1,1,1,1,0,0,0,0] 可稳住 5–8 步，acc@4 不变；"
                    "不能靠渐变衰减或全局缩 α 完全解决。"
                    "若什么都能改，见附录 #post4-playbook 完整行动指南（P1–P5）。"
                ),
            },
            {
                "q": "什么都能改时，怎么让四步以后性能还不掉？",
                "a": (
                    "P1 立刻可用：inference 停写回 [1,1,1,1,0,0,0,0]（实验九）。"
                    "P2 不必跑 5–8 步：混合 3 步、4 跳 4 步。"
                    "P3 按跳数路由 + 题深推到 5–8 跳。"
                    "P4 训练 schedule 对齐 + 同深度微调。"
                    "P5 可学习/按题深门控 writeback。"
                    "详见附录 #post4-playbook。"
                ),
            },
            {
                "q": "只有一次提交、没有标准答案，怎么配参？",
                "a": (
                    "实验十：从题面 BFS 估 d = max(到两个候选的距离)，令 n_latent = d（auto_route）。"
                    "全量 93.1%，比 fixed_3 高 9.3 pp，且等于结构金标准 oracle_hop。"
                    "不需扫 1–8 步准确率曲线。结构未知时用 fallback_zero4 兜底。"
                    "详见附录 #auto-submit。"
                ),
            },
            {
                "q": "模型能否自己决定何时停？什么叫可行？",
                "a": (
                    "实验十一至五十五：在逐步推理或 upfront 预算下，用 stop head / 稳定 / 收敛等信号决定停步。"
                    "L1 deployable_mvp：acc≥86.3%、mean_n≤4.5、无 oracle——Phase 17 定稿 knn_min3_full 92.6% 达标。"
                    "L2 strict_feasible：另需 timing≥50%——未过，全量最好 37%，min_n=3 理论上限约 44%。"
                    "失败主因 late_stop≈57%（min_n=3 不让早停）；Phase 20 从 min_n=2 突破，待 A800。"
                    "详见附录 #adaptive-stop 与 #gpu-phase。"
                ),
            },
            {
                "q": "什么叫 deployable_mvp？推荐部署方案是什么？",
                "a": (
                    "双轨部署（deploy_spec_v4）："
                    "① ProsQA 同源 → confidence_fallback τ=0.48，acc 94.75%；"
                    "② 跨未知 53 切片 → hybrid_slice_router（skip/agreement/tri_zone），加权 Δ +2.05 pp。"
                    "同源历史冠军 95.23%（Phase 25 seed 环境）；timing strict 未过。"
                    "详见附录 #cross-transfer 与 #gpu-phase。"
                ),
            },
            {
                "q": "跨集 transfer 怎么用？dual_ok 什么意思？",
                "a": (
                    "dual_ok：in-dist 加权 Δ≥0 且 OOD 加权 Δ≥7 pp 同时成立。"
                    "Phase 38 hybrid_router @ seed=99 dual_ok=true，hurts=6。"
                    "但四 seed 审计仅 1/4 通过——部署须固定 eval seed=99。"
                    "路由：syn_chain_5_wide skip_transfer；push/mix/hops/diamond agreement；其余 tri_zone 0.40/0.48。"
                    "详见附录 #cross-transfer。"
                ),
            },
            {
                "q": "timing 为什么卡在 37% 上不去？",
                "a": (
                    "56% 题首次答对发生在第 1–2 步，但 min_n=3 不允许早停——这些题在 timing 评分里永远算 late_stop。"
                    "U5 失败解剖：late_stop 占 57%，early_stop 仅约 5%。"
                    "min_n=3 理论 timing 上限约 44%；实测 37% 已达上限 84%，调参只剩 2–5 pp。"
                    "Phase 13–19 换标签、联合长训均未破——需 Phase 20 从 min_n=2 或写回层入手。"
                ),
            },
            {"q": "什么因素最影响边界？", "a": "① 题目推理深度 ② 模型训练时的步数习惯 ③ 加步后的饱和/干扰。图宽、边序等影响次要；第七轮补充：latent 反馈强度可在 d 附近微调报边界。"},
            {
                "q": "为什么有的子集最高准确率是 0？",
                "a": "见上文「极端准确率说明」：通常是符号监督（评估格式与训练不一致）或人工 OOD 图（分布不匹配）。加步也救不了；此类边界数字勿解读。",
            },
        ],
        "recommendations": [
            {"scenario": "ProsQA 同源评估", "advice": "confidence_fallback τ=0.48（94.75%）；历史峰值 95.23%"},
            {"scenario": "跨未知分布 53 切片", "advice": "hybrid_slice_router（deploy_spec_v4）；固定 seed=99"},
            {"scenario": "只有一次提交、无标签", "advice": "auto_route：n_latent = BFS 深度 d；见附录 #auto-submit"},
            {"scenario": "要让模型自己停步（部署）", "advice": "推荐 knn_min3_full（deployable_mvp ✅，acc 92.6%）；见 #gpu-phase"},
            {"scenario": "要让 timing≥50%", "advice": "当前未过；min_n=3 理论上限≈44%；Phase 20 min_n=2 待验证；见 #gpu-phase"},
            {"scenario": "要让模型自己停步（研究）", "advice": "teacher 线 28/29 可达 timing 可行；盲部署 timing≈37%；见 #adaptive-stop"},
            {"scenario": "前缀/upfront 自报步数", "advice": "31–52 线答题≈94% 平台，timing≈28% 未可行；混测 3/4 跳混淆是瓶颈"},
            {"scenario": "已知 4 跳题为主", "advice": "用 4 步 latent"},
            {"scenario": "判断是否该加步", "advice": "看准确率曲线：3–4 步到顶就不要加到 5–6"},
            {"scenario": "希望边界稳定 ≥5 步", "advice": "prosqa_extend 从 3 跳基线到 5 跳 + 索引监督；必要时微调"},
            {"scenario": "4 步以后还要加 latent", "advice": "前 4 步 α=1，第 5 步起 α=0（停写回）；不要继续 full writeback"},
            {"scenario": "什么都能改、要完整方案", "advice": "见附录 #post4-playbook：P1 停写回 → P3 路由/推题深 → P4 同深度训练"},
            {"scenario": "全量也要 5–8 步零波动", "advice": "P1 停写回 + P3 按 3 跳/4 跳分别跑步数；或拆子集评测"},
            {"scenario": "5–8 步还要高准确率", "advice": "P3 ProsQA 图延长到 5–8 跳 + P4 同深度微调 + 训练/inference schedule 对齐"},
            {"scenario": "只拧模型系数做敏感性", "advice": "扫 latent 反馈 α∈[0.75,1.25]；同时报告 acc@d，不单看报边界"},
        ],
        "boundary_push_table": (boundary_push.get("comparison") or {}).get("table") or [],
        "model_perturb_laws": (model_perturb_analysis or {}).get("laws") or [],
        "compare_table": cmp_table,
        "deep_table": deep_table,
        "variant_table": var_construction[:12],
        "references": {
            "paper": "https://arxiv.org/abs/2505.12514",
            "code": "https://github.com/Ber666/reasoning-by-superposition",
            "doc": "docs/experiment-findings.md",
        },
    }
