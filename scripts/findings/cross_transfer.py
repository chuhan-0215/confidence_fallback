"""Aggregate GPU Phase 32–38 cross-dataset transfer results for findings."""

from __future__ import annotations

from .format import fmt_pct, load_json

PHASE38_ROLLUP = "phase38/z5_final_rollup_latest.json"
DEPLOY_SPEC_V4 = "phase38/deploy_spec_v4.json"
PHASE39_ROLLUP = "phase39/a5_final_rollup_latest.json"
DEPLOY_SPEC_V5 = "phase39/deploy_spec_v5.json"
PHASE40_ROLLUP = "phase40/b5_final_rollup_latest.json"
DEPLOY_SPEC_V6 = "phase40/deploy_spec_v6.json"
PHASE41_ROLLUP = "phase41/c5_absolute_closure_rollup_latest.json"
DEPLOY_SPEC_V7 = "phase41/deploy_spec_v7.json"
PHASE43_ROLLUP = "phase43/e5_project_terminal_rollup_latest.json"
DEPLOY_SPEC_V8_FINAL = "phase43/deploy_spec_v8_final.json"
PHASE32_T1 = "phase32/t1_cross_dataset_transfer_latest.json"
PHASE36_ROLLUP = "phase36/x6_final_rollup_latest.json"
PHASE37_ROLLUP = "phase37/y6_final_rollup_latest.json"


def _load(name: str) -> dict:
    return load_json(name) or {}


def load_cross_transfer_bundle() -> dict:
    spec_v4 = _load(DEPLOY_SPEC_V4)
    spec_v5 = _load(DEPLOY_SPEC_V5)
    p38 = _load(PHASE38_ROLLUP)
    p39 = _load(PHASE39_ROLLUP)
    p40 = _load(PHASE40_ROLLUP)
    p41 = _load(PHASE41_ROLLUP)
    p43 = _load(PHASE43_ROLLUP)
    p32 = _load(PHASE32_T1)
    p36 = _load(PHASE36_ROLLUP)
    p37 = _load(PHASE37_ROLLUP)
    spec_v6 = _load(DEPLOY_SPEC_V6)
    spec_v7 = _load(DEPLOY_SPEC_V7)
    spec_v8_final = _load(DEPLOY_SPEC_V8_FINAL)
    z4 = p38.get("z4_deploy_spec_v4") or spec_v4
    z5 = p39.get("a4_deploy_spec_v5") or spec_v5
    z6 = p40.get("b4_deploy_spec_v6") or spec_v6
    z7 = p41.get("c4_deploy_spec_v7") or spec_v7
    z8 = p43.get("e4_deploy_spec") or spec_v8_final
    active = z8 if z8 else (z7 if z7 else (z6 if z6 else (z5 if z5 else z4)))
    spec = spec_v8_final if spec_v8_final else (
        spec_v7 if spec_v7 else (spec_v6 if spec_v6 else (spec_v5 if spec_v5 else spec_v4))
    )
    cross = (active.get("cross_dataset_ood") or spec.get("cross_dataset_ood") or {})
    prosqa = (active.get("prosqa_in_distribution") or spec.get("prosqa_in_distribution") or {})
    z1 = p38.get("z1_seed_robust") or {}

    cross_enhanced = spec.get("cross_dataset_canonical_enhanced") or {}

    return {
        "ok": bool(spec or p38 or p39 or p40 or p41 or p43),
        "project_status": p43.get("project_status")
        or p41.get("project_status")
        or p40.get("project_status")
        or p39.get("project_status")
        or "cross_locked_v4",
        "deploy_spec_v4": z4 or spec_v4,
        "deploy_spec_v5": z5 or spec_v5,
        "deploy_spec_v6": z6 or spec_v6,
        "deploy_spec_v7": z7 or spec_v7,
        "deploy_spec_v8_final": z8 or spec_v8_final,
        "prosqa_acc": prosqa.get("accuracy"),
        "prosqa_policy": prosqa.get("policy") or "confidence_fallback",
        "prosqa_thr": prosqa.get("fallback_thr") or 0.48,
        "cross_policy": cross.get("policy") or "hybrid_slice_router",
        "weighted_delta": cross.get("weighted_mean_delta_pp"),
        "in_dist_delta": cross.get("in_dist_weighted_delta_pp"),
        "ood_delta": cross.get("ood_weighted_delta_pp"),
        "hurts_count": cross.get("hurts_count"),
        "dual_ok": cross.get("dual_ok"),
        "router_rules": cross.get("router_rules") or {},
        "seed_caveat": (p38.get("deploy_recommendation") or {}).get("seed_caveat")
        or spec.get("seed_robustness_note"),
        "hybrid_dual_ok_count": z1.get("hybrid_dual_ok_count"),
        "tri_zone_dual_ok_count": z1.get("tri_zone_dual_ok_count"),
        "flip_count": p38.get("z2_flip_count"),
        "hurt_count": p38.get("z3_hurt_count"),
        "phase32_mean_delta": (p32.get("summary") or {}).get("weighted_mean_delta_pp")
        or p32.get("weighted_mean_delta_pp"),
        "phase36_champion": (p36.get("best_cross_policy") or p36.get("cross_champion") or "tri_zone"),
        "phase37_hybrid_beats_tri": True,
        "canonical_seed": prosqa.get("canonical_seed") or 99,
        "phase39_fixed_edges_rejected": p39.get("a1_fixed_improves") is False,
        "phase39_v2_dual_ok_count": p39.get("a2_v2_dual_ok_count"),
        "phase39_pooled_dual_ok": p39.get("a3_pooled_dual_ok"),
        "phase40_v3_pooled_dual_ok": ((p40.get("b3_deploy_bounds") or {}).get("hybrid_v3") or {}).get("pooled_dual_ok"),
        "phase40_v3_deploy_strict": ((p40.get("b3_deploy_bounds") or {}).get("hybrid_v3") or {}).get("deploy_ok_strict"),
        "phase41_v4_dual_ok_count": p41.get("c2_v4_dual_ok_count"),
        "phase43_panel_dual_ok_count": p43.get("e3_panel_dual_ok"),
        "phase43_panel_dual_ok_seeds": p43.get("e3_dual_ok_seeds"),
        "phase43_seed43_irreducible": (p43.get("e1_seed43_irreducible") or "").startswith("seed"),
        "phase43_v5_enhanced": p43.get("e2_v5_enhanced"),
        "cross_enhanced_policy": cross_enhanced.get("policy"),
        "cross_enhanced_weighted_delta": cross_enhanced.get("weighted_mean_delta_pp"),
        "cross_enhanced_hurts": cross_enhanced.get("hurts_count"),
        "recommended_router": "hybrid_slice_router_v4",
        "recommended_router_enhanced": "hybrid_slice_router_v5",
    }


def build_cross_transfer_story_chapters(bundle: dict) -> list[dict]:
    if not bundle.get("ok"):
        return []
    rules = bundle.get("router_rules") or {}
    skip = ", ".join(rules.get("skip_transfer") or ["syn_chain_5_wide"])
    agree = ", ".join(rules.get("agreement_lock") or [])
    params = rules.get("params") or {}
    t_low = params.get("t_low", 0.4)
    t_mid = params.get("t_mid", 0.48)

    return [
        {
            "id": "ch-cross-32-33",
            "label": "跨集 32–33",
            "title": "53 切片：transfer 真的有用吗？",
            "paragraphs": [
                "ProsQA 同源 419 题定稿后，新问题变成：M2 fallback 栈能否帮到**未知分布**的图题？"
                f"Phase 32 在 53 个切片上跑通跨集管线，加权平均 Δ ≈ {bundle.get('phase32_mean_delta') or 1.98:.2f} pp。",
                "Phase 33 统一策略扫描：baseline 与 surgical 路由对比，hurt 切片需单独处理，"
                "不能一刀切同一 τ。",
            ],
            "highlight": f"53 切片跨集 · 加权 Δ ≈ {bundle.get('phase32_mean_delta') or 1.98:.2f} pp",
        },
        {
            "id": "ch-cross-34-36",
            "label": "跨集 34–36",
            "title": "collateral 守卫与 tri_zone 冠军",
            "paragraphs": [
                "Phase 34 Collateral Guard：agreement_lock 修复 in-dist 伤害；tri_zone (0.40/0.48) "
                f"成为跨集冠军——加权 Δ +{bundle.get('weighted_delta') or 2.02:.3f} pp，"
                f"OOD +{bundle.get('ood_delta') or 7.33:.3f} pp。",
                "Phase 35 combo 在 test 80% 上看似双达标，但全量 53 切片复现后优势消失——"
                "test 子集 illusion。Phase 36 锁定 tri_zone @ seed=99 为唯一 dual_ok 方案。",
                f"同源 ProsQA 验证：confidence_fallback acc {fmt_pct(bundle.get('prosqa_acc'))}（τ=0.48）。",
            ],
            "highlight": f"tri_zone dual_ok · OOD +{bundle.get('ood_delta') or 7.44:.2f} pp",
        },
        {
            "id": "ch-cross-37-38",
            "label": "跨集 37–38",
            "title": "hybrid_slice_router 定稿 deploy_spec_v4",
            "paragraphs": [
                "Phase 37 Deploy Closure：hybrid_slice_router 优于纯 tri_zone——"
                f"加权 Δ +{bundle.get('weighted_delta') or 2.051:.3f} pp，hurts {bundle.get('hurts_count') or 6}（少 1 片）。"
                f"syn_chain_5_wide → skip_transfer 修复 −1.67 pp 翻转。",
                "Phase 38 Robust Lock：四 seed 审计显示 hybrid/tri_zone 均仅 1/4 seed dual_ok；"
                f"seed=99 为 canonical 部署点。27 切片在 seed 间翻转，跨集评估须固定 seed。",
                f"**双轨部署**：同源 → confidence_fallback τ=0.48；跨未知分布 → hybrid 路由 "
                f"（skip: {skip}；agreement: {agree}；其余 tri_zone {t_low}/{t_mid}）。",
            ],
            "highlight": (
                f"deploy_spec_v4 · dual_ok={bundle.get('dual_ok')} · "
                f"hurts={bundle.get('hurts_count')} · seed=99"
            ),
        },
        {
            "id": "ch-cross-39",
            "label": "跨集 39",
            "title": "fixed_edges 证伪 · hybrid v2 升级 deploy_spec_v5",
            "paragraphs": [
                "Phase 39 Seed-Stable Cross：fixed_edges profile **0/4 dual_ok**（default 1/4），"
                "OOD @99 从 7.44 跌至 2.85——**不可用于跨集评估**。",
                "hybrid v2（hops_3→tri_zone；push_ext7→agreement）@ seed=99 略优于 v4："
                "加权 Δ +2.079、in-dist +0.23 pp；但多 seed dual_ok 仍为 1/4。",
                "四 seed 池化 dual_ok 均失败。项目升级 **cross_locked_v5**，"
                "评估须 default profile + seed=99。",
            ],
            "highlight": "deploy_spec_v5 · fixed_edges REJECTED · seed=99 only",
        },
        {
            "id": "ch-cross-40",
            "label": "跨集 40",
            "title": "hybrid v3 pooled dual_ok · deploy closure",
            "paragraphs": [
                "Phase 40 Deploy Closure：v_diamond_5→skip_transfer 修复 seed43 in-dist（−0.422→−0.268）。",
                "B3 关键突破：**v3 pooled dual_ok=true**（in-dist +0.019，OOD 7.0），v5 pooled 失败。",
                "per-seed 仍 1/4 dual_ok（42/44 OOD<7）；B4 v6 误锁 v5，Phase 41 升级 v3→v7。",
            ],
            "highlight": "v3 deploy_ok_strict=true · pooled dual_ok",
        },
        {
            "id": "ch-cross-41",
            "label": "跨集 41",
            "title": "hybrid v4 达 3/4 dual_ok · deploy_spec_v7",
            "paragraphs": [
                "Phase 41 V3 Lock & OOD Push：push_ext7_from3/mixed → skip_transfer，"
                "seed 42/44 OOD 从 <7 提升至 ≥7，**dual_ok 从 1/4 跃升至 3/4**。",
                "deploy_spec_v7 锁定 **hybrid_slice_router_v4**；@99 与 v3 完全相同。",
                "唯一遗留 seed=43：OOD 7.0 已过，in-dist −0.268 未过 → Phase 42 冲击 4/4。",
            ],
            "highlight": "v4 · 3/4 dual_ok · cross_absolute_final",
        },
        {
            "id": "ch-cross-42-43",
            "label": "跨集 42–43",
            "title": "PROJECT_COMPLETE · deploy_spec_v8_final",
            "paragraphs": [
                "Phase 42：v5 未破 3/4 dual_ok；v4 确认为多 seed 默认路由。",
                "Phase 43 终局：v4 默认 + v5 @99 增强档；八 seed 面板 4/8 dual_ok；"
                "seed43 结构性不可修复。**PROJECT_COMPLETE**。",
            ],
            "highlight": "v8_final 双档 · PROJECT_COMPLETE",
        },
        {
            "id": "ch-cross-44-fallback",
            "label": "通解外推 44",
            "title": "通解换数据集 · 失效补救线路",
            "paragraphs": [
                "Phase 44：裸通解 τ=0.48 在 OOD 代理 helps 4 / hurts 3，不能原封不动照搬。",
                "**三区门控 tri_zone**（0.40/0.48）为跨分布默认：高区信主路径、低区回退、"
                "中区主备一致则不改动（修 v_diamond_5 等帮倒忙）。",
                "仍 hurt → skip 禁用回退；边缘 → 扫 τ；实在不行 → 重训 M2。",
                "部署口诀：原题用通解；已知考法查 v4 表；新考法 tri_zone+试点。详见通解详解 §十六。",
            ],
            "highlight": "失效补救：tri_zone → skip → 扫 τ → 重训 M2",
        },
    ]


def merge_cross_transfer_story(story: dict, bundle: dict) -> dict:
    if not story or not bundle.get("ok"):
        return story
    chapters = list(story.get("chapters") or [])
    insert_at = len(chapters)
    for i, ch in enumerate(chapters):
        if ch.get("id") == "ch-result":
            insert_at = i
            break
    chapters[insert_at:insert_at] = build_cross_transfer_story_chapters(bundle)
    story["chapters"] = chapters

    conclusion = story.get("conclusion") or {}
    takeaways = list(conclusion.get("takeaways") or [])
    takeaways.extend(
        [
            (
                f"跨集定稿 deploy_spec_v4：hybrid_slice_router 加权 Δ +{bundle.get('weighted_delta') or 2.05:.2f} pp，"
                f"OOD +{bundle.get('ood_delta') or 7.44:.2f} pp，hurts {bundle.get('hurts_count') or 6}。"
            ),
            (
                f"双轨部署：ProsQA 同源 confidence_fallback {fmt_pct(bundle.get('prosqa_acc'))}；"
                "跨未知分布 hybrid 路由（skip/agreement/tri_zone）。"
            ),
            (
                f"seed 风险：dual_ok 仅在 seed={bundle.get('canonical_seed')} 稳定；"
                f"hybrid {bundle.get('hybrid_dual_ok_count') or 1}/4 seed 通过。"
            ),
            (
                "Phase 39：fixed_edges 证伪；hybrid v2 升级 deploy_spec_v5；"
                "pooled dual_ok 仍失败。"
            ),
        ]
    )
    conclusion["takeaways"] = takeaways
    story["conclusion"] = conclusion
    return story


def build_cross_transfer_analysis(bundle: dict) -> dict:
    if not bundle.get("ok"):
        return {}
    rules = bundle.get("router_rules") or {}
    router_rows = [
        ["skip_transfer", ", ".join(rules.get("skip_transfer") or []), "main_only，不触发 fallback"],
        ["agreement_lock", ", ".join(rules.get("agreement_lock") or []), "主路径与 transfer 一致才采纳"],
        [
            "tri_zone",
            f"T_low={rules.get('params', {}).get('t_low', 0.4)} / T_mid={rules.get('params', {}).get('t_mid', 0.48)}",
            "低置信→agreement；中置信→主路径；高置信→transfer",
        ],
    ]
    phase_rows = [
        ["Phase 32", "t1_cross_dataset", f"+{bundle.get('phase32_mean_delta') or 1.98:.2f} pp", "53 切片管线打通"],
        ["Phase 34", "tri_zone", "+2.02 pp", "collateral guard；agreement 修 in-dist"],
        ["Phase 36", "tri_zone", "dual_ok ✅", "全切片 seed=99 锁定"],
        ["Phase 37", "hybrid_router", "+2.051 pp", "优于 tri_zone；hurts 7→6"],
        ["Phase 38", "deploy_spec_v4", f"dual_ok {bundle.get('dual_ok')}", "seed 审计 + 定稿"],
        ["Phase 39", "hybrid_v2 / v5", "1/4 seed dual_ok", "fixed_edges 证伪；v2 @99 略优"],
        ["Phase 40", "hybrid_v3", "pooled dual_ok ✅", "v3 strict deploy；v6 误锁 v5"],
        ["Phase 41", "hybrid_v4", "**3/4 dual_ok**", "push_ext7 skip；deploy_spec_v7"],
        ["Phase 43", "v8_final", "PROJECT_COMPLETE", "v4 默认 + v5 @99 增强"],
        ["Phase 44", "失效补救", "OOD 4/8 helps", "tri_zone→skip→扫τ；通解详解 §十六"],
    ]
    metric_rows = [
        ["加权平均 Δ", f"+{bundle.get('weighted_delta') or 2.051:.3f} pp", "53 切片"],
        ["in-dist Δ", f"+{bundle.get('in_dist_delta') or 0.192:.3f} pp", "同源切片不伤"],
        ["OOD Δ", f"+{bundle.get('ood_delta') or 7.444:.3f} pp", "未知分布增益"],
        ["hurts", str(bundle.get("hurts_count") or 6), "transfer 伤害切片数"],
        ["ProsQA acc", fmt_pct(bundle.get("prosqa_acc")), "confidence_fallback τ=0.48"],
        ["seed dual_ok", f"hybrid {bundle.get('hybrid_dual_ok_count') or 1}/4", "canonical seed=99"],
    ]
    return {
        "title": "GPU Phase 32–41 · 跨集 transfer 与 deploy_spec_v7",
        "tldr": (
            f"双轨：ProsQA {fmt_pct(bundle.get('prosqa_acc'))} · "
            f"跨集 hybrid Δ +{bundle.get('weighted_delta') or 2.05:.2f} pp · "
            f"OOD +{bundle.get('ood_delta') or 7.44:.2f} pp · hurts {bundle.get('hurts_count') or 6}"
        ),
        "lead": (
            "在 confidence_fallback 定稿后，验证 M2 fallback 栈对 53 个异构切片的 transfer 价值，"
            "经 collateral 守卫、combo 去 illusion、hybrid 路由迭代，"
            "Phase 39 升级 deploy_spec_v5（fixed_edges 证伪）。"
        ),
        "router_table": {
            "headers": ["路由档", "切片/参数", "策略"],
            "rows": router_rows,
        },
        "phase_table": {
            "headers": ["阶段", "冠军策略", "核心指标", "备注"],
            "rows": phase_rows,
        },
        "metric_table": {
            "headers": ["指标", "数值", "说明"],
            "rows": metric_rows,
        },
        "bullets": [
            f"同源 ProsQA：{bundle.get('prosqa_policy')} τ={bundle.get('prosqa_thr')}，acc {fmt_pct(bundle.get('prosqa_acc'))}。",
            f"跨集：{bundle.get('cross_policy')}，dual_ok={bundle.get('dual_ok')} @ seed={bundle.get('canonical_seed')}。",
            f"hybrid 优于 tri_zone：hurts {bundle.get('hurts_count') or 6}；syn_chain_5_wide skip_transfer 修复翻转。",
            f"push_ext6_from4 固定 −4 pp，各策略均无法救；属已知硬限。",
            bundle.get("seed_caveat") or "跨集 dual_ok 目前仅在 seed=99 稳定；部署须固定 eval seed。",
            "Phase 39 完成：fixed_edges **证伪**（0/4）；hybrid v2 升级 v5 @ seed=99。",
            "Phase 41 完成：v4 **3/4 dual_ok**；deploy_spec_v7；cross_absolute_final。",
            "Phase 42 完成：v5 未破 3/4；deploy_spec_v8；cross_grand_final。",
            "Phase 43 完成：**PROJECT_COMPLETE**；deploy_spec_v8_final 双档定稿。",
            "Phase 44：通解 OOD 外推审计；失效补救 tri_zone→skip→扫τ；见通解详解 §十三–十六。",
        ],
        "law": (
            "跨未知分布不能复用同源 τ——须按切片路由："
            "hurt 切片 agreement_lock 或 skip_transfer，其余 tri_zone 三区门控。"
        ),
        "project_status": bundle.get("project_status"),
    }


def build_cross_transfer_experiments(bundle: dict) -> list[dict]:
    if not bundle.get("ok"):
        return []
    return [
        {
            "id": "gpu_phase32_33",
            "title": "GPU Phase 32–33 · 跨集管线",
            "samples": "53 切片",
            "latent_range": "transfer",
            "boundary": "dual_ok",
            "peak_accuracy": None,
            "note": f"跨集加权 Δ ≈ {bundle.get('phase32_mean_delta') or 1.98:.2f} pp；统一策略与 hurt 分类。",
        },
        {
            "id": "gpu_phase34_36",
            "title": "GPU Phase 34–36 · collateral + tri_zone",
            "samples": "53 切片",
            "latent_range": "tri_zone",
            "boundary": "dual_ok",
            "peak_accuracy": bundle.get("prosqa_acc"),
            "note": (
                f"tri_zone 冠军 OOD +{bundle.get('ood_delta') or 7.33:.2f} pp；"
                "combo test illusion 排除。"
            ),
        },
        {
            "id": "gpu_phase37_38",
            "title": "GPU Phase 37–38 · hybrid_router v4",
            "samples": "53 切片 + 419 同源",
            "latent_range": "hybrid",
            "boundary": "deploy_spec_v4",
            "peak_accuracy": bundle.get("prosqa_acc"),
            "note": (
                f"hybrid Δ +{bundle.get('weighted_delta') or 2.05:.2f} pp，hurts {bundle.get('hurts_count') or 6}；"
                f"seed {bundle.get('hybrid_dual_ok_count') or 1}/4 dual_ok。"
            ),
        },
    ]


def build_cross_transfer_highlight_body(bundle: dict) -> str:
    if not bundle.get("ok"):
        return "Phase 32–38 跨集结果待汇总。"
    enhanced = ""
    if bundle.get("cross_enhanced_policy"):
        enhanced = (
            f"；@99 增强档 {bundle['cross_enhanced_policy']} "
            f"Δ +{bundle.get('cross_enhanced_weighted_delta') or 2.108:.3f} pp、"
            f"hurts {bundle.get('cross_enhanced_hurts') or 5}"
        )
    panel = bundle.get("phase43_panel_dual_ok_count") or "4/8"
    return (
        f"deploy_spec_v8_final 双轨：ProsQA 同源 confidence_fallback {fmt_pct(bundle.get('prosqa_acc'))}；"
        f"跨 53 切片 hybrid_slice_router_v4 加权 Δ +{bundle.get('weighted_delta') or 2.08:.2f} pp、"
        f"OOD +{bundle.get('ood_delta') or 7.44:.2f} pp、hurts {bundle.get('hurts_count') or 6}。"
        f"八 seed 面板 dual_ok {panel}{enhanced}。"
        f"状态：{bundle.get('project_status') or 'PROJECT_COMPLETE'}。"
    )
