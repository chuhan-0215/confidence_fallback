"""Render findings sections to HTML fragments."""

from __future__ import annotations

from datetime import datetime

from glossary import GLOSSARY_URL, render_text_with_terms

from .format import esc, fmt_pct, fmt_step

def render_extreme_cases_html(extreme: dict) -> str:
    if not extreme:
        return ""

    parts = [
        '<section class="findings-extreme" id="extreme-accuracy">',
        f'<h3>{esc(extreme.get("title"))}</h3>',
        f'<p class="findings-extreme-tldr">{esc(extreme.get("tldr"))}</p>',
        f'<p class="findings-extreme-def">{esc(extreme.get("definition"))}</p>',
    ]

    rows = extreme.get("rows") or []
    by_cat: dict[str, list] = {}
    for row in rows:
        by_cat.setdefault(row.get("category") or "other", []).append(row)

    for cat in extreme.get("categories") or []:
        cat_rows = by_cat.get(cat.get("id") or "", [])
        parts.append('<article class="findings-extreme-card">')
        parts.append(
            f'<div class="findings-extreme-head">'
            f'<h4>{esc(cat.get("title"))}</h4>'
            f'<span class="extreme-tag">{esc(cat.get("tag"))}</span></div>'
        )
        parts.append(f'<p>{esc(cat.get("why"))}</p>')
        if cat.get("contrast"):
            parts.append(f'<p class="findings-extreme-contrast"><strong>对照：</strong>{esc(cat.get("contrast"))}</p>')
        if cat_rows:
            parts.append('<table class="findings-mechanism-table findings-extreme-table">')
            parts.append(
                "<thead><tr><th>子集</th><th>实验</th><th>监督</th><th>峰值准确率</th><th>检测边界</th></tr></thead><tbody>"
            )
            for row in cat_rows:
                parts.append(
                    f"<tr><td>{esc(row.get('label'))}</td>"
                    f"<td>{esc(row.get('experiment'))}</td>"
                    f"<td>{esc(row.get('supervision'))}</td>"
                    f"<td>{fmt_pct(row.get('max_accuracy'))}</td>"
                    f"<td>{fmt_step(row.get('boundary'))} 步</td></tr>"
                )
            parts.append("</tbody></table>")
        parts.append("</article>")

    parts.append('<div class="findings-extreme-foot">')
    parts.append("<h4>常见误读</h4><ul>")
    for item in extreme.get("misreadings") or []:
        parts.append(f"<li>{esc(item)}</li>")
    parts.append("</ul>")
    if extreme.get("takeaway"):
        parts.append(f'<p class="findings-extreme-takeaway"><strong>总结：</strong>{esc(extreme.get("takeaway"))}</p>')
    parts.append("</div></section>")
    return "\n        ".join(parts)


def _render_story_guide(guide: dict) -> str:
    if not guide:
        return ""
    parts = ['<aside class="story-four-guide" aria-label="故事提要">']
    parts.append(f'<h4>{esc(guide.get("title", "故事提要"))}</h4>')
    parts.append('<div class="story-four-grid">')
    for item in guide.get("items") or []:
        parts.append('<article class="story-four-card">')
        parts.append(f'<div class="story-four-q">{esc(item.get("q"))}</div>')
        parts.append(f'<p class="story-four-a">{render_text_with_terms(item.get("a", ""))}</p>')
        parts.append("</article>")
    parts.append("</div></aside>")
    return "\n".join(parts)


def _render_reader_qa(qa: list) -> str:
    if not qa:
        return ""
    parts = ['<section class="story-reader-qa" aria-label="读者追问">']
    parts.append(f'<h4>{esc(qa[0].get("section_title", "常见追问"))}</h4>')
    parts.append('<dl class="story-reader-qa-list">')
    for item in qa:
        parts.append(f'<dt>{esc(item.get("q"))}</dt>')
        parts.append(f'<dd>{render_text_with_terms(item.get("a", ""))}</dd>')
    parts.append("</dl></section>")
    return "\n".join(parts)


def render_site_guide_html(guide: dict | None) -> str:
    if not guide:
        return ""
    parts = [
        '<nav class="exp-site-guide" aria-label="阅读导航">',
        f'<p class="exp-site-guide-lead">{esc(guide.get("lead", ""))}</p>',
        '<div class="exp-site-guide-grid">',
    ]
    for item in guide.get("items") or []:
        parts.append(
            f'<a class="exp-site-guide-card" href="{esc(item.get("href", "#"))}">'
            f'<span class="exp-site-guide-tag">{esc(item.get("tag", ""))}</span>'
            f'<strong>{esc(item.get("title", ""))}</strong>'
            f'<span>{esc(item.get("desc", ""))}</span>'
            "</a>"
        )
    parts.append("</div></nav>")
    return "\n        ".join(parts)


def render_story_toc_html(chapters: list) -> str:
    if not chapters:
        return ""
    parts = [
        '<nav class="story-toc" aria-label="章节目录">',
        '<div class="story-toc-title">章节目录</div>',
        '<ol class="story-toc-list">',
    ]
    for ch in chapters:
        parts.append(
            f'<li><a href="#{esc(ch.get("id", ""))}">'
            f'<span class="story-toc-label">{esc(ch.get("label", ""))}</span>'
            f'{render_text_with_terms(ch.get("title", ""))}</a></li>'
        )
    parts.append("</ol></nav>")
    return "\n        ".join(parts)


def render_appendix_toc_html() -> str:
    links = [
        ("essence-laws", "本质与普遍规律"),
        ("pattern-laws", "实验五 · 规律寻探"),
        ("push-ladder", "实验六–八 · 边界上推"),
        ("feedback-schedule", "实验九 · 数据表"),
        ("post4-playbook", "实验九 · 行动指南"),
        ("auto-submit", "实验十 · 通解验证"),
        ("adaptive-stop", "实验十一–五十五 · 自停探索"),
        ("gpu-phase", "GPU Phase 16–20 · 全量定稿"),
        ("cross-transfer", "GPU Phase 32–38 · 跨集 deploy_spec_v4"),
        ("model-perturb", "实验七 · 模型扰动"),
        ("appendix-reference", "FAQ · 建议 · 十轮卡片"),
    ]
    parts = [
        '<nav class="appendix-toc" id="appendix-toc" aria-label="附录目录">',
        '<h3 class="appendix-toc-title">附录目录</h3>',
        '<div class="appendix-toc-grid">',
    ]
    for anchor, label in links:
        parts.append(f'<a class="appendix-toc-link" href="#{esc(anchor)}">{esc(label)}</a>')
    parts.append("</div></nav>")
    return "\n        ".join(parts)


def render_story_html(story: dict) -> str:
    if not story:
        return ""

    parts = [
        '<section class="findings-story" id="story-guide">',
        '<header class="story-header">',
        f'<span class="story-badge">{esc(story.get("badge"))}</span>',
        f'<h3>{esc(story.get("title"))}</h3>',
        f'<p class="story-subtitle">{esc(story.get("subtitle"))}</p>',
    ]
    if story.get("hook"):
        parts.append(f'<p class="story-hook">{render_text_with_terms(story.get("hook"))}</p>')
    if story.get("guide"):
        parts.append(_render_story_guide(story.get("guide") or {}))
    parts.append(
        f'<p class="story-glossary-hint">文中带<span class="term-ref-demo">下划线</span>的词可查阅'
        f'<a href="{GLOSSARY_URL}">术语注释</a>。</p>'
    )
    parts.append("</header>")

    chapters = story.get("chapters") or []
    if chapters:
        parts.append(render_story_toc_html(chapters))

    parts.append('<div class="story-chapters">')

    for ch in story.get("chapters") or []:
        parts.append(f'<article class="story-chapter" id="{esc(ch.get("id"))}">')
        parts.append(
            f'<div class="story-chapter-label">{esc(ch.get("label"))}</div>'
            f'<h4>{esc(ch.get("title"))}</h4>'
        )
        for para in ch.get("paragraphs") or []:
            parts.append(f'<p>{render_text_with_terms(para)}</p>')
        if ch.get("highlight"):
            parts.append(f'<p class="story-highlight">{render_text_with_terms(ch.get("highlight"))}</p>')
        parts.append("</article>")

    parts.append("</div>")

    if story.get("reader_qa"):
        parts.append(_render_reader_qa(story.get("reader_qa") or []))

    conclusion = story.get("conclusion") or {}
    if conclusion:
        parts.append('<article class="story-conclusion">')
        parts.append(f'<h4>{esc(conclusion.get("title"))}</h4>')
        for para in conclusion.get("paragraphs") or []:
            parts.append(f'<p>{render_text_with_terms(para)}</p>')
        takeaways = conclusion.get("takeaways") or []
        if takeaways:
            parts.append('<ol class="story-takeaways">')
            for t in takeaways:
                parts.append(f"<li>{render_text_with_terms(t)}</li>")
            parts.append("</ol>")
        if conclusion.get("formula"):
            parts.append(f'<p class="story-formula">{esc(conclusion.get("formula"))}</p>')
        parts.append("</article>")

    science = story.get("science_box") or {}
    if science:
        parts.append('<aside class="story-science">')
        parts.append(f'<h4>{esc(science.get("title"))}</h4>')
        if science.get("experiment"):
            parts.append(f'<p class="story-science-exp">{render_text_with_terms(science.get("experiment"))}</p>')
        laws = science.get("laws") or []
        if laws:
            parts.append('<details class="story-science-laws-wrap">')
            parts.append(f'<summary>十二条规律（点击展开）</summary>')
            parts.append('<ul class="story-science-laws">')
            for law in laws:
                parts.append(f"<li>{render_text_with_terms(law)}</li>")
            parts.append("</ul></details>")
        stats = science.get("stats") or []
        if stats:
            parts.append('<div class="story-science-stats">')
            for label, value in stats:
                parts.append(
                    f'<div class="story-science-stat">'
                    f'<strong>{render_text_with_terms(value)}</strong><span>{esc(label)}</span></div>'
                )
            parts.append("</div>")
        push_callout = science.get("push_callout") or {}
        if push_callout:
            parts.append('<div class="story-push-callout">')
            parts.append(f'<h5>{esc(push_callout.get("title"))}</h5>')
            if push_callout.get("formula"):
                parts.append(
                    f'<p class="story-push-formula">{render_text_with_terms(push_callout.get("formula"))}</p>'
                )
            if push_callout.get("detail_href"):
                parts.append(
                    f'<a class="story-push-more" href="{esc(push_callout.get("detail_href"))}">'
                    "完整上推对照表 ↓</a>"
                )
            parts.append("</div>")
        parts.append("</aside>")

    if story.get("appendix_link"):
        parts.append(f'<p class="story-appendix-link">{story.get("appendix_link")}</p>')

    parts.append("</section>")
    return "\n        ".join(parts)


def render_essence_html(essence: dict) -> str:
    if not essence:
        return ""

    parts = [
        '<section class="findings-essence" id="essence-laws">',
        f'<h3>{esc(essence.get("title"))}</h3>',
        f'<p class="essence-core">{esc(essence.get("core_mechanism"))}</p>',
        f'<p class="essence-universal"><strong>普遍规律：</strong>{render_text_with_terms(essence.get("universal_law"))}</p>',
    ]
    if essence.get("push_formula"):
        parts.append(
            f'<div class="essence-push-formula" id="push-formula">'
            f'<strong>边界上推公式</strong>'
            f'<p>{render_text_with_terms(essence.get("push_formula"))}</p></div>'
        )
    push_conditions = essence.get("push_conditions") or []
    if push_conditions:
        parts.append('<div class="essence-push-conditions">')
        parts.append("<h4>三必要条件（缺一即失败或过冲）</h4>")
        parts.append('<div class="essence-push-conditions-grid">')
        for cond in push_conditions:
            parts.append('<article class="essence-push-condition">')
            parts.append(f'<h5>{esc(cond.get("title"))}</h5>')
            parts.append(f'<p>{esc(cond.get("text"))}</p>')
            if cond.get("fail"):
                parts.append(f'<p class="essence-push-fail">{esc(cond.get("fail"))}</p>')
            parts.append("</article>")
        parts.append("</div></div>")
    push_ladder = essence.get("push_ladder") or {}
    ladder_rows = push_ladder.get("rows") or []
    if ladder_rows:
        parts.append(f'<div class="essence-push-ladder" id="push-ladder">')
        parts.append(f'<h4>{esc(push_ladder.get("title", "边界上推对照表"))}</h4>')
        parts.append('<table class="findings-mechanism-table essence-push-table"><thead><tr>')
        for h in push_ladder.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in ladder_rows:
            parts.append("<tr>" + "".join(f"<td>{render_text_with_terms(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table></div>")
    antipatterns = essence.get("push_antipatterns") or []
    if antipatterns:
        parts.append('<div class="essence-push-antipatterns">')
        parts.append("<h4>什么做不到（反模式）</h4><ul>")
        for pt in antipatterns:
            parts.append(f"<li>{render_text_with_terms(pt)}</li>")
        parts.append("</ul></div>")
    parts.extend([
        '<div class="essence-compare">',
    ])

    compare = essence.get("compare") or {}
    for side_key, css in (("left", "essence-col essence-col--34"), ("right", "essence-col essence-col--56")):
        side = compare.get(side_key) or {}
        parts.append(f'<article class="{css}">')
        parts.append(f'<h4>{esc(side.get("label"))}</h4>')
        if side.get("summary"):
            parts.append(f'<p class="essence-col-summary">{esc(side.get("summary"))}</p>')
        if side.get("points"):
            parts.append("<ul>")
            for pt in side.get("points") or []:
                parts.append(f"<li>{esc(pt)}</li>")
            parts.append("</ul>")
        if side.get("when_yes"):
            parts.append(f'<p class="essence-subhead essence-subhead--yes">{esc(side.get("when_yes_label") or "可行路径")}</p><ul>')
            for pt in side.get("when_yes") or []:
                parts.append(f'<li class="essence-yes">{esc(pt)}</li>')
            parts.append("</ul>")
        if side.get("when_no"):
            parts.append(f'<p class="essence-subhead essence-subhead--no">{esc(side.get("when_no_label") or "失败与反模式")}</p><ul>')
            for pt in side.get("when_no") or []:
                parts.append(f'<li class="essence-no">{esc(pt)}</li>')
            parts.append("</ul>")
        parts.append("</article>")

    parts.append("</div>")

    laws = essence.get("laws") or []
    if laws:
        parts.append('<div class="essence-laws-grid">')
        for law in laws:
            parts.append(
                f'<article class="essence-law-card">'
                f'<h5>{esc(law.get("title"))}</h5>'
                f'<p>{esc(law.get("text"))}</p></article>'
            )
        parts.append("</div>")

    if essence.get("conclusion"):
        parts.append(
            f'<div class="essence-conclusion">'
            f'<strong>总结结论</strong>'
            f'<p>{esc(essence.get("conclusion"))}</p></div>'
        )

    parts.append("</section>")
    return "\n        ".join(parts)


def render_tldr_html(mech: dict) -> str:
    tldr = mech.get("tldr") or mech.get("conclusion")
    if not tldr:
        return ""
    return "\n        ".join(
        [
            '<aside class="findings-tldr" id="why-3-4-steps">',
            '<div class="findings-tldr-label">一句话</div>',
            f'<p class="findings-tldr-text">{render_text_with_terms(tldr)}</p>',
            '<a class="findings-tldr-more" href="#mech-expand">展开说明 ↓</a>',
            "</aside>",
        ]
    )


MECH_BLOCK_ANCHORS = ["mech-bfs", "mech-ablation", "mech-mixed", "mech-ceiling"]


def render_mechanism_html(mech: dict) -> str:
    if not mech:
        return ""

    parts = [
        '<section class="findings-mechanism" id="mech-expand">',
        f'<h3>{esc(mech.get("title"))}<span class="findings-mechanism-sub">展开说明</span></h3>',
        f'<p class="findings-mechanism-lead">{render_text_with_terms(mech.get("lead"))}</p>',
    ]

    blocks = mech.get("blocks") or []
    if blocks:
        parts.append('<nav class="findings-mechanism-nav" aria-label="展开说明目录">')
        for idx, block in enumerate(blocks):
            anchor = MECH_BLOCK_ANCHORS[idx] if idx < len(MECH_BLOCK_ANCHORS) else f"mech-block-{idx + 1}"
            label = (block.get("heading") or "").split(".", 1)[-1].strip() or block.get("heading")
            parts.append(f'<a href="#{anchor}">{esc(label)}</a>')
        if mech.get("caveats", {}).get("rows"):
            parts.append('<a href="#mech-caveats">适用范围</a>')
        parts.append("</nav>")

    for idx, block in enumerate(blocks):
        anchor = MECH_BLOCK_ANCHORS[idx] if idx < len(MECH_BLOCK_ANCHORS) else f"mech-block-{idx + 1}"
        parts.append(f'<div class="findings-mechanism-block" id="{anchor}">')
        parts.append(f'<h4>{esc(block.get("heading"))}</h4>')
        for para in block.get("paragraphs") or []:
            parts.append(f'<p>{render_text_with_terms(para)}</p>')
        bullets = block.get("bullets") or []
        if bullets:
            parts.append("<ul>")
            for b in bullets:
                parts.append(f"<li>{render_text_with_terms(b)}</li>")
            parts.append("</ul>")
        table = block.get("table")
        if table:
            parts.append('<table class="findings-mechanism-table">')
            parts.append("<thead><tr>")
            for h in table.get("headers") or []:
                parts.append(f"<th>{esc(h)}</th>")
            parts.append("</tr></thead><tbody>")
            for row in table.get("rows") or []:
                parts.append("<tr>")
                for cell in row:
                    parts.append(f"<td>{esc(cell)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table>")
        if block.get("footnote"):
            parts.append(f'<p class="findings-mechanism-note">{esc(block.get("footnote"))}</p>')
        parts.append("</div>")

    caveats = mech.get("caveats") or {}
    if caveats:
        parts.append('<div class="findings-mechanism-block" id="mech-caveats">')
        parts.append(f'<h4>{esc(caveats.get("heading"))}</h4>')
        parts.append('<table class="findings-mechanism-table">')
        parts.append("<thead><tr><th>情况</th><th>现象</th><th>说明</th></tr></thead><tbody>")
        for row in caveats.get("rows") or []:
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{esc(cell)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

    chain = mech.get("causal_chain") or []
    if chain:
        parts.append('<div class="findings-mechanism-block findings-causal">')
        parts.append("<h4>因果链</h4><ol>")
        for step in chain:
            parts.append(f"<li>{esc(step)}</li>")
        parts.append("</ol></div>")

    tiers = mech.get("evidence_tiers") or []
    if tiers:
        parts.append('<div class="findings-mechanism-block">')
        parts.append("<h4>依据可靠程度</h4>")
        parts.append('<table class="findings-mechanism-table findings-tier-table">')
        parts.append("<thead><tr><th>强度</th><th>依据</th><th>结论</th></tr></thead><tbody>")
        for t in tiers:
            parts.append(
                f'<tr class="tier-{esc(t.get("tier"))}">'
                f'<td><span class="tier-badge">{esc(t.get("tier"))}</span></td>'
                f'<td>{esc(t.get("basis"))}</td>'
                f'<td>{esc(t.get("conclusion"))}</td></tr>'
            )
        parts.append("</tbody></table></div>")

    parts.append("</section>")
    return "\n        ".join(parts)


def render_pattern_laws_html(laws_payload: dict) -> str:
    if not laws_payload or not laws_payload.get("laws"):
        return ""

    parts = [
        '<section class="findings-patterns" id="pattern-laws">',
        "<h3>实验五 · 六条可复现规律</h3>",
        f'<p class="findings-patterns-lead">{esc(laws_payload.get("unified_conclusion"))}</p>',
    ]

    corr = laws_payload.get("correlations") or {}
    if corr.get("boundary_vs_mean_hops") is not None:
        parts.append(
            '<p class="findings-patterns-meta">'
            f'边界↔平均跳数 r={corr.get("boundary_vs_mean_hops")} · '
            f'边界↔图直径 r={corr.get("boundary_vs_mean_diameter")}'
            "</p>"
        )

    for law in laws_payload.get("laws") or []:
        tier = law.get("evidence_tier", "")
        parts.append('<article class="pattern-law-card">')
        parts.append(
            f'<div class="pattern-law-head"><h4>{esc(law.get("title"))}</h4>'
            f'<span class="tier-badge">{esc(tier)}</span></div>'
        )
        parts.append(f'<p class="pattern-obs"><strong>观察到的规律：</strong>{esc(law.get("pattern"))}</p>')
        parts.append(f'<p class="pattern-why"><strong>为什么：</strong>{esc(law.get("why"))}</p>')
        examples = law.get("examples") or []
        if examples:
            parts.append('<ul class="pattern-examples">')
            for ex in examples:
                parts.append(
                    f'<li><strong>{esc(ex.get("label"))}</strong>{esc(ex.get("value"))}</li>'
                )
            parts.append("</ul>")
        parts.append("</article>")

    mix = laws_payload.get("mix_ladder") or {}
    if mix.get("available") and mix.get("rows"):
        parts.append('<div class="findings-mechanism-block"><h4>混合比例阶梯（实验五）</h4>')
        parts.append(f'<p>{esc(mix.get("summary"))}</p>')
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        parts.append("<th>4跳占比</th><th>检测边界</th><th>峰值准确率</th></tr></thead><tbody>")
        for row in mix["rows"]:
            parts.append(
                f'<tr><td>{row.get("ratio_4hop_pct")}%</td>'
                f'<td>{fmt_step(row.get("boundary"))} 步</td>'
                f'<td>{fmt_pct(row.get("max_accuracy"))}</td></tr>'
            )
        parts.append("</tbody></table></div>")

    cross = laws_payload.get("hop_diameter_cross") or {}
    if cross.get("available") and cross.get("rows"):
        parts.append('<div class="findings-mechanism-block"><h4>跳数×直径交叉（实验五）</h4>')
        parts.append(f'<p>{esc(cross.get("summary"))}</p>')
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        parts.append("<th>子集</th><th>平均跳数</th><th>平均直径</th><th>边界</th></tr></thead><tbody>")
        for row in cross["rows"]:
            parts.append(
                f'<tr><td>{esc(row.get("label"))}</td>'
                f'<td>{esc(row.get("mean_hops"))}</td>'
                f'<td>{esc(row.get("mean_diameter"))}</td>'
                f'<td>{fmt_step(row.get("boundary"))} 步</td></tr>'
            )
        parts.append("</tbody></table></div>")

    parts.append("</section>")
    return "\n        ".join(parts)


def render_boundary_push_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-push" id="boundary-push">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    table = section.get("table") or {}
    rows = table.get("rows") or []
    if rows:
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")
    elif section.get("table_href"):
        parts.append(
            f'<p class="muted">明细对照表见 <a href="{esc(section.get("table_href"))}">'
            f'{esc(section.get("table_link_label") or "边界上推对照表")}</a>。</p>'
        )
    for bullet in section.get("bullets") or []:
        if bullet:
            parts.append(f"<p>{render_text_with_terms(bullet)}</p>")
    if section.get("law"):
        parts.append(f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("law"))}</p>')
    if section.get("push_formula"):
        parts.append(
            f'<div class="essence-push-formula essence-push-formula--inline">'
            f'<strong>普遍上推公式</strong>'
            f'<p>{render_text_with_terms(section.get("push_formula"))}</p></div>'
        )
    parts.append("</section>")
    return "\n        ".join(parts)


def render_boundary_push_deep_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-push findings-push-deep" id="boundary-push-deep">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    table = section.get("table") or {}
    rows = table.get("rows") or []
    if rows:
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")
    elif section.get("table_href"):
        parts.append(
            f'<p class="muted">明细对照表见 <a href="{esc(section.get("table_href"))}">'
            f'{esc(section.get("table_link_label") or "边界上推对照表")}</a>。</p>'
        )
    for bullet in section.get("bullets") or []:
        if bullet:
            parts.append(f"<p>{render_text_with_terms(bullet)}</p>")
    if section.get("law"):
        parts.append(f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("law"))}</p>')
    parts.append("</section>")
    return "\n        ".join(parts)


def render_feedback_schedule_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-feedback-schedule" id="feedback-schedule">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    if section.get("mechanism"):
        parts.append(f'<p class="muted">{render_text_with_terms(section.get("mechanism"))}</p>')
    table = section.get("table") or {}
    rows = table.get("rows") or []
    if rows:
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")
    bullets = section.get("bullets") or []
    if bullets:
        parts.append("<ul>")
        for b in bullets:
            if b:
                parts.append(f"<li>{render_text_with_terms(b)}</li>")
        parts.append("</ul>")
    parts.append("</section>")
    return "\n        ".join(parts)


def render_feedback_playbook_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-feedback-playbook" id="post4-playbook">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("intro"):
        parts.append(f'<p>{render_text_with_terms(section.get("intro"))}</p>')
    if section.get("metrics_note"):
        parts.append(f'<p class="muted">{render_text_with_terms(section.get("metrics_note"))}</p>')

    result_table = section.get("result_table") or {}
    result_rows = result_table.get("rows") or []
    if result_rows:
        parts.append("<h4>停写回效果一览（实验九数据）</h4>")
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in result_table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in result_rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")

    tiers = section.get("tiers") or []
    if tiers:
        parts.append("<h4>方案优先级（P1 → P5）</h4>")
        parts.append('<div class="feedback-playbook-tiers">')
        for tier in tiers:
            parts.append('<article class="feedback-playbook-tier essence-push-condition">')
            parts.append(
                f'<div class="feedback-playbook-priority">{esc(tier.get("priority", ""))}</div>'
            )
            parts.append(f'<h5>{render_text_with_terms(tier.get("title", ""))}</h5>')
            if tier.get("when"):
                parts.append(
                    f'<p><strong>适用：</strong>{render_text_with_terms(tier.get("when"))}</p>'
                )
            if tier.get("how"):
                parts.append(
                    f'<p><strong>做法：</strong>{render_text_with_terms(tier.get("how"))}</p>'
                )
            if tier.get("result"):
                parts.append(
                    f'<p><strong>效果：</strong>{render_text_with_terms(tier.get("result"))}</p>'
                )
            if tier.get("limit"):
                parts.append(
                    f'<p class="essence-push-fail"><strong>局限：</strong>'
                    f'{render_text_with_terms(tier.get("limit"))}</p>'
                )
            parts.append("</article>")
        parts.append("</div>")

    avoid = section.get("avoid_table") or {}
    avoid_rows = avoid.get("rows") or []
    if avoid_rows:
        parts.append('<div class="essence-push-antipatterns">')
        parts.append("<h4>不推荐的做法</h4>")
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in avoid.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in avoid_rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table></div>")

    if section.get("edge_case"):
        parts.append(
            f'<p class="findings-extreme-contrast"><strong>边角说明：</strong>'
            f'{render_text_with_terms(section.get("edge_case"))}</p>'
        )

    decision = section.get("decision") or []
    if decision:
        parts.append("<h4>怎么选？</h4><ul>")
        for item in decision:
            if item:
                parts.append(f"<li>{render_text_with_terms(item)}</li>")
        parts.append("</ul>")

    if section.get("conclusion"):
        parts.append(
            f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("conclusion"))}</p>'
        )
    parts.append("</section>")
    return "\n        ".join(parts)


def render_auto_submit_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-auto-submit" id="auto-submit">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    table = section.get("table") or {}
    rows = table.get("rows") or []
    if rows:
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")
    recipe = section.get("recipe") or {}
    if recipe:
        parts.append(f'<h4>{render_text_with_terms(recipe.get("title", ""))}</h4>')
        steps = recipe.get("steps") or []
        if steps:
            parts.append("<ol>")
            for step in steps:
                if step:
                    parts.append(f"<li>{render_text_with_terms(step)}</li>")
            parts.append("</ol>")
        if recipe.get("code"):
            parts.append(
                f'<p class="muted"><strong>实现：</strong>{esc(recipe.get("code"))}</p>'
            )
    bullets = section.get("bullets") or []
    if bullets:
        parts.append("<ul>")
        for b in bullets:
            if b:
                parts.append(f"<li>{render_text_with_terms(b)}</li>")
        parts.append("</ul>")
    if section.get("law"):
        parts.append(
            f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("law"))}</p>'
        )
    parts.append("</section>")
    return "\n        ".join(parts)


def render_adaptive_stop_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-adaptive-stop" id="adaptive-stop">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    table = section.get("table") or {}
    rows = table.get("rows") or []
    if rows:
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")
    if section.get("table_note"):
        parts.append(f'<p class="muted">{esc(section.get("table_note"))}</p>')
    bullets = section.get("bullets") or []
    if bullets:
        parts.append("<ul>")
        for b in bullets:
            if b:
                parts.append(f"<li>{render_text_with_terms(b)}</li>")
        parts.append("</ul>")
    if section.get("law"):
        parts.append(
            f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("law"))}</p>'
        )
    parts.append("</section>")
    return "\n        ".join(parts)


def render_gpu_phase_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-gpu-phase" id="gpu-phase">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')

    for key, title in (
        ("proof_table", "五层证明（Phase 17 定稿）"),
        ("phase_table", "Phase 16–19 关键方案"),
        ("failure_table", "U5 失败类型分解"),
    ):
        table = section.get(key) or {}
        rows = table.get("rows") or []
        if rows:
            parts.append(f"<h4>{esc(title)}</h4>")
            parts.append('<table class="findings-mechanism-table"><thead><tr>')
            for h in table.get("headers") or []:
                parts.append(f"<th>{esc(h)}</th>")
            parts.append("</tr></thead><tbody>")
            for row in rows:
                parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
            parts.append("</tbody></table>")

    if section.get("ceiling_insight"):
        parts.append(
            f'<p class="findings-mechanism-foot"><strong>timing 天花板：</strong>'
            f'{render_text_with_terms(section.get("ceiling_insight"))}</p>'
        )
    bullets = section.get("bullets") or []
    if bullets:
        parts.append("<ul>")
        for b in bullets:
            if b:
                parts.append(f"<li>{render_text_with_terms(b)}</li>")
        parts.append("</ul>")
    if section.get("law"):
        parts.append(
            f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("law"))}</p>'
        )
    if section.get("phase20_pending"):
        parts.append(
            '<p class="muted">Phase 20（V1–V5）已设计，待 A800 运行；'
            '详见 <a href="lab.html#labCatGpu">交互实验室 · GPU 批次</a>。</p>'
        )
    parts.append("</section>")
    return "\n        ".join(parts)


def render_cross_transfer_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-cross-transfer" id="cross-transfer">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    for key, title in (
        ("router_table", "hybrid_slice_router 路由规则（v4）"),
        ("phase_table", "Phase 32–38 里程碑"),
        ("metric_table", "跨集核心指标 @ seed=99"),
    ):
        table = section.get(key) or {}
        rows = table.get("rows") or []
        if rows:
            parts.append(f"<h4>{esc(title)}</h4>")
            parts.append('<table class="findings-mechanism-table"><thead><tr>')
            for h in table.get("headers") or []:
                parts.append(f"<th>{esc(h)}</th>")
            parts.append("</tr></thead><tbody>")
            for row in rows:
                parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
            parts.append("</tbody></table>")
    bullets = section.get("bullets") or []
    if bullets:
        parts.append("<ul>")
        for b in bullets:
            if b:
                parts.append(f"<li>{render_text_with_terms(b)}</li>")
        parts.append("</ul>")
    if section.get("law"):
        parts.append(
            f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("law"))}</p>'
        )
    if section.get("project_status"):
        parts.append(
            f'<p class="muted">项目状态：<code>{esc(section.get("project_status"))}</code> · '
            f'定稿 <code>results/phase41/deploy_spec_v7.json</code></p>'
        )
    parts.append("</section>")
    return "\n        ".join(parts)


def render_model_perturb_html(section: dict) -> str:
    if not section:
        return ""
    parts = [
        '<section class="findings-perturb" id="model-perturb">',
        f'<h3>{render_text_with_terms(section.get("title", ""))}</h3>',
        f'<p class="findings-mechanism-tldr">{render_text_with_terms(section.get("tldr", ""))}</p>',
    ]
    if section.get("lead"):
        parts.append(f'<p>{render_text_with_terms(section.get("lead"))}</p>')
    if section.get("mechanism"):
        parts.append(f'<p>{render_text_with_terms(section.get("mechanism"))}</p>')
    table = section.get("table") or {}
    rows = table.get("rows") or []
    if rows:
        parts.append('<table class="findings-mechanism-table"><thead><tr>')
        for h in table.get("headers") or []:
            parts.append(f"<th>{esc(h)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</tbody></table>")
    highlights = section.get("highlights") or []
    if highlights:
        parts.append("<ul>")
        for h in highlights:
            if h:
                parts.append(f"<li>{render_text_with_terms(h)}</li>")
        parts.append("</ul>")
    laws = section.get("laws") or []
    if laws:
        parts.append('<div class="findings-pattern-laws">')
        for law in laws:
            parts.append('<article class="pattern-law-card">')
            parts.append(f'<h4>{render_text_with_terms(law.get("title", ""))}</h4>')
            if law.get("pattern"):
                parts.append(
                    f'<p class="pattern-obs"><strong>观察：</strong>'
                    f'{render_text_with_terms(law.get("pattern"))}</p>'
                )
            if law.get("why"):
                parts.append(
                    f'<p class="pattern-why"><strong>为什么：</strong>'
                    f'{render_text_with_terms(law.get("why"))}</p>'
                )
            parts.append("</article>")
        parts.append("</div>")
    if section.get("conclusion"):
        parts.append(f'<p class="findings-mechanism-foot">{render_text_with_terms(section.get("conclusion"))}</p>')
    parts.append("</section>")
    return "\n        ".join(parts)


def _format_generated_at(gen: str) -> str:
    try:
        return datetime.fromisoformat(gen.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return gen


def _collect_appendix_parts(data: dict) -> list[str]:
    appendix: list[str] = [render_appendix_toc_html()]

    essence = data.get("essence") or {}
    if essence:
        appendix.append(render_essence_html(essence))

    laws_payload = data.get("pattern_laws") or {}
    if laws_payload.get("laws"):
        appendix.append(render_pattern_laws_html(laws_payload))

    mech = data.get("mechanism_analysis") or {}
    mech_blocks: list[str] = []
    if mech:
        mech_blocks.append(render_tldr_html(mech))
        mech_blocks.append(render_mechanism_html(mech))
    if mech_blocks:
        appendix.append(
            '<details class="findings-details appendix-mechanism-wrap">'
            '<summary>机制推导与规律详述（可选阅读）</summary>'
            + "\n        ".join(mech_blocks)
            + "</details>"
        )

    extreme = data.get("extreme_cases") or {}
    if extreme:
        appendix.append(render_extreme_cases_html(extreme))

    push = data.get("boundary_push") or {}
    if push:
        appendix.append(render_boundary_push_html(push))

    push_deep = data.get("boundary_push_deep") or {}
    if push_deep:
        appendix.append(render_boundary_push_deep_html(push_deep))

    fb_sched = data.get("feedback_schedule") or {}
    playbook = data.get("feedback_playbook") or {}
    if fb_sched or playbook:
        exp9_parts = ['<div class="appendix-exp9-group" id="appendix-exp9">']
        if fb_sched:
            exp9_parts.append(render_feedback_schedule_html(fb_sched))
        if playbook:
            exp9_parts.append(render_feedback_playbook_html(playbook))
        exp9_parts.append("</div>")
        appendix.append("\n        ".join(exp9_parts))

    auto_submit = data.get("auto_submit") or {}
    if auto_submit:
        appendix.append(render_auto_submit_html(auto_submit))

    adaptive_stop = data.get("adaptive_stop") or {}
    if adaptive_stop:
        appendix.append(render_adaptive_stop_html(adaptive_stop))

    gpu_phase = data.get("gpu_phase") or {}
    if gpu_phase:
        appendix.append(render_gpu_phase_html(gpu_phase))

    cross_transfer = data.get("cross_transfer") or {}
    if cross_transfer:
        appendix.append(render_cross_transfer_html(cross_transfer))

    perturb = data.get("model_perturb") or {}
    if perturb:
        appendix.append(render_model_perturb_html(perturb))

    why = data.get("why_analysis") or {}
    supplemental: list[str] = ['<section class="appendix-reference" id="appendix-reference">']
    supplemental.append("<h3>速查与明细</h3>")

    if why.get("summary") or why.get("reasons") or why.get("marginal_gains"):
        supplemental.append('<details class="findings-details">')
        supplemental.append("<summary>数据驱动的简要归因</summary>")
        supplemental.append('<div class="findings-why">')
        if why.get("summary"):
            supplemental.append(f'<p class="findings-why-summary">{esc(why.get("summary"))}</p>')
        reasons = why.get("reasons") or []
        if reasons:
            supplemental.append('<ul class="why-list">')
            for r in reasons:
                supplemental.append(
                    f"<li><strong>{esc(r.get('title'))}</strong>"
                    f"<span>{esc(r.get('detail'))}</span></li>"
                )
            supplemental.append("</ul>")
        gains = why.get("marginal_gains") or []
        if gains:
            supplemental.append(
                '<table class="findings-marginal"><thead><tr>'
                "<th>步数</th><th>准确率</th><th>相对上一步</th></tr></thead><tbody>"
            )
            for g in gains:
                delta = g.get("delta_pct_points")
                delta_s = f"{delta:+.1f}pp" if delta is not None else "—"
                supplemental.append(
                    f"<tr><td>{esc(g.get('n_latent'))}</td>"
                    f"<td>{fmt_pct(g.get('accuracy'))}</td>"
                    f"<td>{esc(delta_s)}</td></tr>"
                )
            supplemental.append("</tbody></table>")
        supplemental.append("</div></details>")

    highlights = data.get("highlights") or []
    if highlights:
        supplemental.append('<details class="findings-details">')
        supplemental.append("<summary>跨实验要点速览</summary>")
        supplemental.append('<ul class="findings-highlights">')
        for h in highlights:
            supplemental.append(
            f"<li><strong>{esc(h.get('title'))}</strong>{render_text_with_terms(h.get('body', ''))}</li>"
        )
        supplemental.append("</ul></details>")

    experiments = data.get("experiments") or []
    if experiments:
        supplemental.append('<details class="findings-details">')
        supplemental.append("<summary>实验卡片（边界 + 自停）</summary>")
        supplemental.append('<div class="findings-grid">')
        for e in experiments:
            boundary = e.get("boundary")
            if isinstance(boundary, (int, float)):
                boundary = f"{fmt_step(boundary)} 步"
            supplemental.append(
                f'<article class="findings-exp-card"><h3>{esc(e.get("title"))}</h3>'
                f'<div class="meta">{esc(e.get("samples"))} · 扫描 {esc(e.get("latent_range"))}</div>'
                f'<div class="boundary">{esc(boundary)}</div>'
                f'<div class="meta">峰值 {fmt_pct(e.get("peak_accuracy"))}</div>'
                f'<p>{esc(e.get("note"))}</p></article>'
            )
        supplemental.append("</div></details>")

    insights = data.get("insights") or []
    if insights:
        supplemental.append('<details class="findings-details">')
        supplemental.append("<summary>跨实验对照语句</summary>")
        supplemental.append('<div class="findings-insights"><ul>')
        for line in insights:
            supplemental.append(f"<li>{esc(line)}</li>")
        supplemental.append("</ul></div></details>")

    supplemental.append('<details class="findings-details" open><summary>核心问题 FAQ</summary>')
    supplemental.append('<dl class="findings-faq">')
    for f in data.get("faq") or []:
        supplemental.append(f"<dt>{esc(f.get('q'))}</dt><dd>{esc(f.get('a'))}</dd>")
    supplemental.append("</dl></details>")

    supplemental.append('<details class="findings-details" open><summary>实用建议</summary>')
    supplemental.append('<table class="findings-rec-table"><thead><tr><th>场景</th><th>建议</th></tr></thead><tbody>')
    for r in data.get("recommendations") or []:
        supplemental.append(f"<tr><td>{esc(r.get('scenario'))}</td><td>{esc(r.get('advice'))}</td></tr>")
    supplemental.append("</tbody></table></details>")

    supplemental.append('<details class="findings-details"><summary>多数据集对比明细（实验二）</summary>')
    supplemental.append('<div class="findings-table-wrap"><table class="compare-table findings-table">')
    supplemental.append("<thead><tr><th>子集</th><th>平均跳数</th><th>边界</th><th>峰值准确率</th></tr></thead><tbody>")
    for row in data.get("compare_table") or []:
        supplemental.append(
            f"<tr><td><strong>{esc(row.get('label'))}</strong></td>"
            f"<td>{esc(row.get('mean_reasoning_hops'))}</td>"
            f"<td>{fmt_step(row.get('boundary'))} 步</td>"
            f"<td>{fmt_pct(row.get('max_accuracy'))}</td></tr>"
        )
    supplemental.append("</tbody></table></div></details>")

    push_table = data.get("boundary_push_table") or []
    if push_table:
        supplemental.append('<details class="findings-details"><summary>边界上推明细（实验六）</summary>')
        supplemental.append('<div class="findings-table-wrap"><table class="compare-table findings-table">')
        supplemental.append(
            "<thead><tr><th>构造</th><th>平均跳数</th><th>边界</th><th>峰值准确率</th></tr></thead><tbody>"
        )
        for row in push_table:
            supplemental.append(
                f"<tr><td><strong>{esc(row.get('label'))}</strong></td>"
                f"<td>{esc(row.get('mean_reasoning_hops'))}</td>"
                f"<td>{fmt_step(row.get('boundary'))} 步</td>"
                f"<td>{fmt_pct(row.get('max_accuracy'))}</td></tr>"
            )
        supplemental.append("</tbody></table></div></details>")

    supplemental.append('<details class="findings-details"><summary>构造×监督对照要点（实验四）</summary>')
    supplemental.append('<div class="findings-table-wrap"><table class="compare-table findings-table">')
    supplemental.append("<thead><tr><th>子集</th><th>构造</th><th>边界</th><th>峰值准确率</th></tr></thead><tbody>")
    for row in data.get("variant_table") or []:
        supplemental.append(
            f"<tr><td>{esc(row.get('label'))}</td>"
            f"<td>{esc(row.get('construction'))}</td>"
            f"<td>{fmt_step(row.get('boundary'))} 步</td>"
            f"<td>{fmt_pct(row.get('max_accuracy'))}</td></tr>"
        )
    supplemental.append("</tbody></table></div></details>")

    supplemental.append("</section>")
    appendix.extend(supplemental)
    return appendix


def render_appendix_html(data: dict) -> str:
    appendix = _collect_appendix_parts(data)
    if not appendix:
        return '<p class="findings-footnote">暂无附录内容。</p>'

    refs = data.get("references") or {}
    parts = list(appendix)
    parts.append(
        '<p class="findings-footnote">完整报告见 <code>docs/experiment-findings.md</code> · '
        f'论文 <a href="{esc(refs.get("paper", ""))}" target="_blank" rel="noopener">'
        "Reasoning by Superposition (NeurIPS 2025)</a></p>"
    )
    return "\n        ".join(parts)


def render_math_proof_html(proof: dict) -> str:
    if not proof:
        return ""

    parts = [
        '<section class="findings-proof" id="math-proof">',
        f'<h3>{esc(proof.get("title"))}</h3>',
    ]
    if proof.get("intro"):
        parts.append(f'<p class="findings-proof-intro">{render_text_with_terms(proof.get("intro"))}</p>')

    guide = proof.get("guide") or {}
    if guide:
        parts.append('<div class="proof-guide">')
        parts.append(f'<h4>{esc(guide.get("title", "读前导览"))}</h4>')
        for para in guide.get("paragraphs") or []:
            parts.append(f'<p>{render_text_with_terms(para)}</p>')
        items = guide.get("items") or []
        if items:
            parts.append("<ul>")
            for item in items:
                parts.append(f"<li>{render_text_with_terms(item)}</li>")
            parts.append("</ul>")
        parts.append("</div>")

    if proof.get("theorem"):
        parts.append('<div class="proof-theorem">')
        parts.append('<div class="proof-theorem-label">定理（陈述）</div>')
        parts.append(f'<p class="proof-theorem-text">{render_text_with_terms(proof.get("theorem"))}</p>')
        parts.append("</div>")

    axioms = proof.get("axioms") or []
    if axioms:
        parts.append('<div class="proof-axioms">')
        parts.append('<h4>公理（证明起点）</h4><ul>')
        for ax in axioms:
            parts.append(f"<li>{render_text_with_terms(ax)}</li>")
        parts.append("</ul></div>")

    parts.append('<ol class="proof-steps">')
    for step in proof.get("steps") or []:
        parts.append(f'<li class="proof-step" id="{esc(step.get("id"))}">')
        parts.append('<div class="proof-step-head">')
        parts.append(f'<span class="proof-step-label">{esc(step.get("label"))}</span>')
        parts.append(f'<h4>{esc(step.get("title"))}</h4>')
        parts.append("</div>")
        if step.get("lead"):
            parts.append(f'<p class="proof-step-lead">{render_text_with_terms(step.get("lead"))}</p>')
        math_lines = step.get("math") or []
        if isinstance(math_lines, str):
            math_lines = [math_lines]
        if math_lines:
            parts.append('<div class="proof-math" role="math">')
            for line in math_lines:
                parts.append(f'<div class="proof-math-line">{esc(line)}</div>')
            parts.append("</div>")
        notes = step.get("note")
        if notes:
            if isinstance(notes, str):
                notes = [notes]
            parts.append('<div class="proof-note">')
            parts.append('<span class="proof-note-label">注释</span>')
            for note_para in notes:
                parts.append(f"<p>{render_text_with_terms(note_para)}</p>")
            parts.append("</div>")
        if step.get("example"):
            parts.append('<div class="proof-example">')
            parts.append('<span class="proof-example-label">打个比方</span>')
            parts.append(f'<p>{render_text_with_terms(step.get("example"))}</p>')
            parts.append("</div>")
        parts.append("</li>")
    parts.append("</ol>")

    aside = proof.get("empirical_aside") or {}
    if aside:
        parts.append('<aside class="proof-empirical-aside">')
        parts.append(f'<h4>{esc(aside.get("title"))}</h4>')
        for para in aside.get("paragraphs") or []:
            parts.append(f"<p>{render_text_with_terms(para)}</p>")
        if aside.get("link"):
            parts.append(
                f'<p class="proof-empirical-link">'
                f'<a href="{esc(aside.get("link"))}">{esc(aside.get("link_label", "详见附录"))}</a></p>'
            )
        parts.append("</aside>")

    refs = proof.get("references") or []
    if refs:
        parts.append('<div class="proof-refs">')
        parts.append("<h4>依据</h4><ul>")
        for ref in refs:
            parts.append(f"<li>{esc(ref)}</li>")
        parts.append("</ul></div>")

    parts.append("</section>")
    return "\n        ".join(parts)


def render_findings_html(data: dict) -> str:
    gen_local = _format_generated_at(data.get("generated_at", ""))

    parts = [
        f'<p class="findings-updated">实验已完成 · 汇总于 {esc(gen_local)} · {esc(data.get("model", ""))}</p>',
        '<div class="findings-hero findings-hero--story">',
        f'<p class="findings-headline">{esc(data.get("headline"))}</p>',
        f'<p class="findings-oneliner">{esc(data.get("one_liner"))}</p>',
        "</div>",
    ]

    guide = data.get("site_guide") or {}
    if guide:
        parts.append(render_site_guide_html(guide))

    story = data.get("story") or {}
    if story:
        parts.append(render_story_html(story))

    proof = data.get("math_proof") or {}
    if proof:
        parts.append(render_math_proof_html(proof))

    return "\n        ".join(parts)

