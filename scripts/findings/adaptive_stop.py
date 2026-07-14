"""Aggregate adaptive-stop experiments (11–55) for findings payload."""

from __future__ import annotations

from glossary import GLOSSARY_URL

from .format import fmt_pct, load_json

LEGACY_FILES = {
    11: "adaptive_stop_latest.json",
    12: "adaptive_stop_v2_latest.json",
    13: "adaptive_stop_v3_latest.json",
    14: "adaptive_stop_v4_latest.json",
    15: "adaptive_stop_v5_latest.json",
}

FEASIBILITY_RULE = (
    "可行判定：test 答题 acc ≥ fixed_3（86.3%）且停步时机 stop_timing_acc ≥ 50%。"
)


def _main_acc(summary: dict) -> float | None:
    for key in (
        "main_strategy_accuracy",
        "joint_correctness_stop_accuracy",
        "correctness_stop_accuracy",
        "rich_stop_accuracy",
        "trained_stop_v2_accuracy",
        "trained_stop_accuracy",
    ):
        v = summary.get(key)
        if v is not None:
            return float(v)
    return None


def _main_timing(summary: dict) -> float | None:
    for key in (
        "main_strategy_timing_acc",
        "joint_correctness_stop_timing_acc",
        "correctness_stop_timing_acc",
        "rich_stop_timing_acc",
        "trained_stop_v2_timing_acc",
        "trained_stop_timing_acc",
    ):
        v = summary.get(key)
        if v is not None:
            return float(v)
    return None


def _is_full_eval(data: dict) -> bool:
    n = data.get("sample_count")
    return n in (419, 168)


def _track_status(track: int) -> dict:
    return load_json(f"adaptive_stop_v{track}_status.json") or {}


def _track_running(track: int, data: dict | None, status: dict) -> bool:
    if status.get("running") and status.get("phase") not in ("done", "error", None):
        return True
    if data and not _is_full_eval(data):
        return True
    return False


def load_track_result(exp_id: int) -> dict | None:
    if exp_id in LEGACY_FILES:
        return load_json(LEGACY_FILES[exp_id])
    if 16 <= exp_id <= 55:
        return load_json(f"adaptive_stop_v{exp_id}_latest.json")
    return None


def load_adaptive_stop_bundle() -> dict:
    tracks: list[dict] = []
    baselines: dict[str, float | None] = {
        "fixed_3": None,
        "auto_route": None,
        "oracle_first_correct": None,
    }

    for exp_id in list(range(11, 16)) + list(range(16, 56)):
        data = load_track_result(exp_id)
        if not data or not data.get("ok"):
            continue
        summary = data.get("summary") or {}
        acc = _main_acc(summary)
        timing = _main_timing(summary)
        for key in baselines:
            v = summary.get(f"{key}_accuracy")
            if v is not None and baselines[key] is None:
                baselines[key] = float(v)
        status = _track_status(exp_id) if exp_id >= 16 else {}
        tracks.append(
            {
                "id": exp_id,
                "title": data.get("title") or f"实验{exp_id}",
                "direction": summary.get("direction") or data.get("method_note", "")[:40],
                "method_note": data.get("method_note", ""),
                "accuracy": acc,
                "timing_acc": timing,
                "feasible": summary.get("trainable_stop_feasible"),
                "best_strategy": summary.get("best_learned_strategy"),
                "sample_count": data.get("sample_count"),
                "full_eval": _is_full_eval(data),
                "running": _track_running(exp_id, data, status),
                "phase": status.get("phase") or ("done" if _is_full_eval(data) else "smoke"),
            }
        )

    full_tracks = [t for t in tracks if t["full_eval"] and not t["running"]]
    feasible = [t for t in full_tracks if t["feasible"]]
    blind_candidates = [
        t
        for t in full_tracks
        if not t["feasible"] and t["accuracy"] is not None and t["id"] >= 16
    ]
    best_blind = max(blind_candidates, key=lambda t: (t["accuracy"] or 0, t["timing_acc"] or 0), default=None)
    upfront_candidates = [
        t
        for t in full_tracks
        if t["id"] >= 31 and t["accuracy"] is not None and not t["feasible"]
    ]
    best_upfront = max(upfront_candidates, key=lambda t: t["accuracy"] or 0, default=None)
    best_feasible = max(feasible, key=lambda t: (t["accuracy"] or 0, t["timing_acc"] or 0), default=None)

    legacy = {t["id"]: t for t in tracks if t["id"] <= 15}
    online = [t for t in tracks if t["id"] >= 53]
    online_done = [t for t in online if t["full_eval"] and not t["running"]]

    return {
        "tracks": tracks,
        "full_count": len(full_tracks),
        "baselines": baselines,
        "legacy": legacy,
        "feasible_tracks": feasible,
        "best_blind": best_blind,
        "best_upfront": best_upfront,
        "best_feasible": best_feasible,
        "online": online,
        "online_pending": any(t["running"] or not t["full_eval"] for t in online),
        "online_done": online_done,
        "exp14": legacy.get(14),
    }


def build_adaptive_stop_highlight_body(bundle: dict) -> str:
    b = bundle.get("baselines") or {}
    f3 = fmt_pct(b.get("fixed_3"))
    ar = fmt_pct(b.get("auto_route"))
    oracle = fmt_pct(b.get("oracle_first_correct"))
    best_blind = bundle.get("best_blind") or {}
    best_upfront = bundle.get("best_upfront") or {}
    parts = [
        f"实验十一至五十五问：模型能否自己决定何时停？"
        f"基线 fixed_3 {f3}、auto_route {ar}、oracle 上界 {oracle}。"
        f"{FEASIBILITY_RULE}",
    ]
    if best_blind.get("accuracy") is not None:
        parts.append(
            f"无标签可部署最优：实验{best_blind['id']} {fmt_pct(best_blind['accuracy'])}"
            f"（timing {fmt_pct(best_blind.get('timing_acc'))}）——仍低于 auto_route。"
        )
    if best_upfront.get("accuracy") is not None:
        parts.append(
            f"前缀/upfront 线平台约 {fmt_pct(best_upfront['accuracy'])}"
            f"（timing {fmt_pct(best_upfront.get('timing_acc'))}），未达可行。"
        )
    feasible = bundle.get("feasible_tracks") or []
    if feasible:
        ids = "、".join(str(t["id"]) for t in feasible[:4])
        parts.append(
            f"实验 {ids} 等用首次答对步（teacher）标签可达可行（≈95%），"
            "属 teacher 上界，非盲部署。"
        )
    if bundle.get("online_pending"):
        parts.append("实验五十三至五十五（在线 head∨稳定/收敛）全量评估进行中。")
    return " ".join(parts)


def build_adaptive_stop_story_chapters(bundle: dict, auto_ar: str | None) -> list[dict]:
    b = bundle.get("baselines") or {}
    f3 = fmt_pct(b.get("fixed_3"))
    ar = fmt_pct(b.get("auto_route") or (0.9286 if not auto_ar else None))
    if auto_ar:
        ar = auto_ar
    oracle = fmt_pct(b.get("oracle_first_correct"))
    legacy = bundle.get("legacy") or {}
    e11 = legacy.get(11) or {}
    e14 = legacy.get(14) or {}
    best_blind = bundle.get("best_blind") or {}
    best_upfront = bundle.get("best_upfront") or {}
    best_feas = bundle.get("best_feasible") or {}

    chapters = [
        {
            "id": "ch-adaptive-11",
            "label": "实验十一",
            "title": "新问题：模型能不能自己喊停？",
            "paragraphs": [
                "前十轮解决了「该给几层」——实验十用题面 BFS 路由，一次提交 93.1%。"
                "下一问更刁：不给标准步数、不扫 1–8 步曲线，模型能否在推理过程中自己决定何时停？",
                f"实验十一训 LatentStopHead：用 train 集「首次答对步」作标签，冻结 Coconut，只在 test 评估。"
                f"结果 trained_stop 仅 {fmt_pct(e11.get('accuracy'))}，"
                f"[[stop-timing-acc|停步时机]] {fmt_pct(e11.get('timing_acc'))}——"
                f"远低于 fixed_3 的 {f3}，更够不着 auto_route 的 {ar}。",
                "head 在 train 上过拟合（recall≈100%），test 却普遍停太晚（mean_n≈5.7）。"
                "结论：单靠 hidden 训 stop head，尚不能认定「训练自停」可行。",
            ],
            "highlight": f"trained_stop {fmt_pct(e11.get('accuracy'))} · timing {fmt_pct(e11.get('timing_acc'))} → 不可行",
        },
        {
            "id": "ch-adaptive-12-15",
            "label": "实验十二–十五",
            "title": "换特征、换损失、甚至联合微调——仍不够",
            "paragraphs": [
                "实验十二加正则与 hybrid 停步；实验十三 RichStopHead（答案桶+稳定 streak+focal）；"
                "实验十四改训 is_correct 标签；实验十五解冻 Coconut 最后两层联合微调。"
                f"最好的一档是实验十四 correctness_stop：{fmt_pct(e14.get('accuracy'))}，"
                f"timing {fmt_pct(e14.get('timing_acc'))}——答题率上来了，"
                "但「停在对的步」仍只有约三成。",
                f"实验十五 joint_correctness_stop 80.4%，timing 35%——联合微调也没跨过可行线。"
                "四线并进说明：缺的不是某一个 trick，而是停步信号与 ProsQA 混测结构之间的鸿沟。",
            ],
            "highlight": f"correctness_stop {fmt_pct(e14.get('accuracy'))} · 可行线 timing≥50% 未达",
        },
        {
            "id": "ch-adaptive-16-30",
            "label": "实验十六–三十",
            "title": "四十条并行线：结构、teacher 与 timing 的三角",
            "paragraphs": [
                "实验十六起开并行线：结构深度下界、BFS+Δ 残差、first_correct 标签、"
                "收敛/稳定 OR 门控、BFS 下界+Exp14/15 头等，共十五条线（十六–三十）。",
                f"无 teacher 标签的启发式最好：实验{best_blind.get('id', 24)} "
                f"{fmt_pct(best_blind.get('accuracy'))}（timing {fmt_pct(best_blind.get('timing_acc'))}）——"
                f"超过 auto_route 的答题率，但 timing 仍≈29%，[[trainable-stop-feasible|不可行]]。",
                f"若允许 [[oracle-first-correct|首次答对步]] 作停步标签（teacher 上界），"
                f"实验{best_feas.get('id', 28)}/{best_feas.get('id', 29)} 可达 "
                f"{fmt_pct(best_feas.get('accuracy'))}、timing {fmt_pct(best_feas.get('timing_acc'))}——"
                "可行，但部署时需要答案监督，不是盲停。",
                f"oracle 上界 {oracle} 告诉我们：混测里「停对步且答对」还有约 "
                f"{round((b.get('oracle_first_correct') or 0.9762) * 100 - (b.get('auto_route') or 0.9286) * 100, 1)} pp 空间，"
                "但结构路由已吃掉大部分。",
            ],
            "highlight": (
                f"盲部署 {fmt_pct(best_blind.get('accuracy'))} · teacher 可行 {fmt_pct(best_feas.get('accuracy'))}"
                if best_blind and best_feas
                else "并行线见附录 #adaptive-stop"
            ),
        },
        {
            "id": "ch-adaptive-31-52",
            "label": "实验三十一–五十二",
            "title": "前缀预算：模型自报 n，但 timing 卡住",
            "paragraphs": [
                "实验三十一起转 [[upfront-budget|upfront 预算]]：推理前先决定 n_latent（前缀分类、"
                "图特征 Δ、Coconut 前缀、kNN 集成等），多数策略 test 只需 1–2 次 forward。",
                f"二十余条线（三十一–五十二）答题率平台在 "
                f"{fmt_pct(best_upfront.get('accuracy'))} 附近——"
                f"高于 auto_route 约 {round(((best_upfront.get('accuracy') or 0.94) - (b.get('auto_route') or 0.9286)) * 100, 1)} pp，"
                f"但 timing 卡在 {fmt_pct(best_upfront.get('timing_acc'))}，远低于 50% 可行线。",
                "原因直观：混测里 3 跳与 4 跳题各半，模型常把 4 跳题也预算成 3 步——"
                "答题还行（3 步对 3 跳题仍高），「停步时机」统计却对不上 d。",
                "实验五十二保守回退（P(n=3)<阈值→n=d 而非 n=4）与实验三十三 graph Δ 同为 94.0% 平台——"
                "说明前缀线已触顶，继续堆集成难补 timing 缺口。",
            ],
            "highlight": (
                f"前缀平台 {fmt_pct(best_upfront.get('accuracy'))} · timing {fmt_pct(best_upfront.get('timing_acc'))}"
                if best_upfront
                else "前缀线见附录"
            ),
        },
    ]

    online_pending = bundle.get("online_pending")
    online_done = bundle.get("online_done") or []
    if online_pending:
        online_note = "实验五十三至五十五（在线 RichStopHead∨答案稳定∨hidden 收敛）全量评估进行中。"
        online_highlight = "在线自停 · 全量评估进行中"
    elif online_done:
        best_on = max(online_done, key=lambda t: t["accuracy"] or 0)
        online_note = (
            f"实验五十三–五十五改 [[online-self-stop|在线自停]]：逐步推理，"
            f"head∨稳定∨收敛组合，无 BFS、无 upfront。"
            f"当前最好实验{best_on['id']} {fmt_pct(best_on.get('accuracy'))} "
            f"（timing {fmt_pct(best_on.get('timing_acc'))}）。"
        )
        online_highlight = f"在线线 {fmt_pct(best_on.get('accuracy'))} · timing {fmt_pct(best_on.get('timing_acc'))}"
    else:
        online_note = (
            "实验五十三至五十五探索纯在线自停：逐步推理中 head∨稳定∨收敛，"
            "不依赖 BFS 深度也不 upfront 预算。"
        )
        online_highlight = "在线自停 · 见实验室 G 类"

    chapters.append(
        {
            "id": "ch-adaptive-53-55",
            "label": "实验五十三–五十五",
            "title": "回到逐步推理：在线组合信号",
            "paragraphs": [
                "前缀线触顶后，最后三条线回到逐步 latent："
                "五十三 head∨答案稳定 streak；五十四 head∨hidden 收敛；五十五三者 OR。",
                online_note,
                "与实验十对照：BFS 路由是「先量深度再搜」；在线自停是「边搜边决定停不停」。"
                "若在线线仍够不着 timing≥50%，说明 ProsQA 混测下盲停的核心瓶颈在「分辨 3 跳 vs 4 跳」，"
                "而非 stop head 架构 alone。",
            ],
            "highlight": online_highlight,
        }
    )
    return chapters


def merge_adaptive_stop_story(story: dict, bundle: dict, auto_ar: str | None = None) -> dict:
    if not story or not bundle.get("tracks"):
        return story
    chapters = list(story.get("chapters") or [])
    insert_at = next((i for i, ch in enumerate(chapters) if ch.get("id") == "ch-result"), len(chapters))
    adaptive_chapters = build_adaptive_stop_story_chapters(bundle, auto_ar)
    chapters[insert_at:insert_at] = adaptive_chapters
    story["chapters"] = chapters

    b = bundle.get("baselines") or {}
    ar = auto_ar or fmt_pct(b.get("auto_route"))
    best_blind = bundle.get("best_blind") or {}
    conclusion = story.get("conclusion") or {}
    takeaways = list(conclusion.get("takeaways") or [])
    takeaways.extend(
        [
            (
                f"训练自停（十一–十五）未过可行线；correctness_stop 最好 {fmt_pct((bundle.get('exp14') or {}).get('accuracy'))}。"
            ),
            (
                f"并行线盲部署最高实验{best_blind.get('id', 24)} {fmt_pct(best_blind.get('accuracy'))}；"
                f"teacher 标签下实验 28/29 可达 ≈95% 且可行。"
            ),
            (
                f"前缀/upfront 线（31–52）平台 ≈{fmt_pct((bundle.get('best_upfront') or {}).get('accuracy'))}，"
                "timing≈28–30%，未可行；在线自停 53–55 仍在验证。"
                if bundle.get("online_pending")
                else f"前缀线平台 ≈{fmt_pct((bundle.get('best_upfront') or {}).get('accuracy'))}；在线自停见实验 53–55。"
            ),
            f"混测盲停仍难 beat [[auto-route|auto_route]] {ar}——结构路由是强 baseline。",
        ]
    )
    conclusion["takeaways"] = takeaways
    story["conclusion"] = conclusion

    science = story.get("science_box") or {}
    laws = list(science.get("laws") or [])
    laws.extend(
        [
            (
                f"规律十四：[[trainable-stop-feasible|训练自停可行]]需 acc≥fixed_3 且 timing≥50%；"
                f"十一–十五均未达标；teacher 线 28/29 可达 ≈95%。"
            ),
            (
                f"规律十五：[[upfront-budget|前缀预算]]线答题率≈94%，timing≈28%——"
                "混测 3/4 跳混淆是 timing 瓶颈。"
            ),
            (
                f"规律十六：无标签部署仍应优先 [[auto-route|auto_route]]（{ar}）；"
                "自停线答题可超路由，但停步时机难同时达标。"
            ),
        ]
    )
    science["laws"] = laws
    stats = list(science.get("stats") or [])
    stats.append(("自停盲部署", fmt_pct(best_blind.get("accuracy"))))
    stats.append(("前缀平台", fmt_pct((bundle.get("best_upfront") or {}).get("accuracy"))))
    science["stats"] = stats
    story["science_box"] = science

    result_ch = next((ch for ch in story["chapters"] if ch.get("id") == "ch-result"), None)
    if result_ch:
        paras = list(result_ch.get("paragraphs") or [])
        if paras:
            paras[0] = (
                paras[0]
                + f" 实验十一至五十五进一步问：模型能否自停？"
                f"盲部署最好 {fmt_pct(best_blind.get('accuracy'))}，仍低于 auto_route {ar}；"
                f"前缀线平台 {fmt_pct((bundle.get('best_upfront') or {}).get('accuracy'))}，timing 未过 50%。"
            )
            result_ch["paragraphs"] = paras
    return story


def build_adaptive_stop_analysis(bundle: dict) -> dict:
    if not bundle.get("tracks"):
        return {}
    b = bundle.get("baselines") or {}
    rows = []
    for t in bundle["tracks"]:
        if not t.get("full_eval") and not t.get("running"):
            continue
        status = "进行中" if t.get("running") or not t.get("full_eval") else (
            "可行" if t.get("feasible") else "未可行"
        )
        rows.append(
            [
                f"实验{t['id']}",
                (t.get("direction") or "")[:28],
                fmt_pct(t.get("accuracy")),
                fmt_pct(t.get("timing_acc")),
                status,
            ]
        )

    best_blind = bundle.get("best_blind") or {}
    best_upfront = bundle.get("best_upfront") or {}
    best_feas = bundle.get("best_feasible") or {}
    feasible_ids = "、".join(str(t["id"]) for t in (bundle.get("feasible_tracks") or []))

    bullets = [
        FEASIBILITY_RULE,
        f"test 切分 168 题 · fixed_3 {fmt_pct(b.get('fixed_3'))} · auto_route {fmt_pct(b.get('auto_route'))} · oracle {fmt_pct(b.get('oracle_first_correct'))}。",
        f"训练自停（11–15）：均未可行；Exp14 correctness_stop {fmt_pct((bundle.get('exp14') or {}).get('accuracy'))} 为早期最好。",
        f"并行线（16–30）盲部署最优：实验{best_blind.get('id')} {fmt_pct(best_blind.get('accuracy'))}（timing {fmt_pct(best_blind.get('timing_acc'))}）。",
        f"teacher 可行线：实验 {feasible_ids}，最高 {fmt_pct(best_feas.get('accuracy'))}（timing {fmt_pct(best_feas.get('timing_acc'))}）。",
        f"前缀/upfront（31–52）平台：实验{best_upfront.get('id')} {fmt_pct(best_upfront.get('accuracy'))}（timing {fmt_pct(best_upfront.get('timing_acc'))}）。",
        "在线自停（53–55）：逐步 head∨稳定/收敛；全量结果见实验室 G 类。",
        "部署建议：无标签仍用 auto_route；自停线可研究 timing，但尚未替代 BFS 路由。",
    ]

    tldr = (
        f"45 线自停探索：盲部署 {fmt_pct(best_blind.get('accuracy'))} · "
        f"前缀平台 {fmt_pct(best_upfront.get('accuracy'))} · "
        f"teacher 可行 {fmt_pct(best_feas.get('accuracy'))} · "
        f"基线 auto_route {fmt_pct(b.get('auto_route'))}"
    )

    return {
        "title": "实验十一–五十五 · 自适应停步（模型自停）",
        "tldr": tldr,
        "lead": (
            "在实验十「按题 BFS 给 depth」之后，追问模型能否在推理过程中自己决定何时停。"
            "共 45 条实验线（5 条基础 + 40 条并行/track），统一在 test 168 题上对比 fixed_3、auto_route 与 oracle。"
        ),
        "table": {
            "headers": ["实验", "方向", "答题 acc", "timing", "判定"],
            "rows": rows[:24],
        },
        "table_note": "完整 45 线见实验室 adaptive-stop 总览；上表列出代表性结果。",
        "bullets": bullets,
        "law": (
            "ProsQA 混测下，[[auto-route|auto_route]] 仍是无标签强 baseline；"
            "自停要同时 beat 86.3% 答题与 50% timing 才算 [[trainable-stop-feasible|可行]]。"
        ),
        "feasible_track_ids": [t["id"] for t in (bundle.get("feasible_tracks") or [])],
        "best_blind_track": best_blind.get("id"),
        "best_upfront_track": best_upfront.get("id"),
    }


def build_adaptive_stop_experiments(bundle: dict) -> list[dict]:
    items = [
        {
            "id": "adaptive_stop_arc",
            "title": "实验十一–十五 · 训练自停基础线",
            "samples": "419 · test 168",
            "latent_range": "逐步 1–8",
            "boundary": "可行判定",
            "peak_accuracy": (bundle.get("exp14") or {}).get("accuracy"),
            "note": "LatentStopHead / RichHead / correctness / 联合微调；均未达 timing≥50%。",
        },
        {
            "id": "adaptive_stop_parallel",
            "title": "实验十六–三十 · 并行探索",
            "samples": "15 tracks",
            "latent_range": "逐步+结构",
            "boundary": "4 条可行(teacher)",
            "peak_accuracy": (bundle.get("best_feasible") or {}).get("accuracy"),
            "note": "结构下界、first_correct、BFS+Δ；盲部署最高≈94.6%，teacher 可行≈95.2%。",
        },
        {
            "id": "adaptive_stop_upfront",
            "title": "实验三十一–五十二 · 前缀/upfront 预算",
            "samples": "22 tracks",
            "latent_range": " upfront n",
            "boundary": "timing 瓶颈",
            "peak_accuracy": (bundle.get("best_upfront") or {}).get("accuracy"),
            "note": "答题≈94%，timing≈28–30%；graph Δ / 前缀集成 / 保守回退触顶。",
        },
        {
            "id": "adaptive_stop_online",
            "title": "实验五十三–五十五 · 在线自停",
            "samples": "3 tracks",
            "latent_range": "逐步 OR 组合",
            "boundary": "进行中" if bundle.get("online_pending") else "见结果",
            "peak_accuracy": (
                max((t.get("accuracy") or 0 for t in (bundle.get("online_done") or [])), default=None)
                if not bundle.get("online_pending")
                else None
            ),
            "note": "head∨stable / head∨conv / 三者 OR；无 BFS、无 upfront。",
        },
    ]
    return items
