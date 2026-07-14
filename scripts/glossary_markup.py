"""Apply [[term-id|label]] links to experiment report text."""

from __future__ import annotations

import re

# Longest phrases first; only terms that appear in this experiment's report.
PHRASE_MARKUPS: list[tuple[str, str]] = [
    ("Reasoning by Superposition", "[[neurips-paper|Reasoning by Superposition]]"),
    ("latent ", "[[latent-steps|latent]]"),
    ("BFS", "[[parallel-bfs|BFS]]"),
    ("Fronti", "[[search-frontier|Fronti]]"),
    ("frontier", "[[search-frontier|frontier]]"),
    ("checkpoint_300", "[[checkpoint|checkpoint_300]]"),
    ("checkpoint", "[[checkpoint|checkpoint]]"),
    ("在真实题上延长推理链", "[[prosqa-extend|在真实题上延长推理链]]"),
    ("人工拼出来的长链题", "[[synthetic-chain|人工拼出来的长链题]]"),
    ("6 跳人造链", "[[synthetic-chain|6 跳人造链]]"),
    ("未揭晓的格子", "[[latent-exp|未揭晓的格子]]"),
    ("连续思维步数", "[[latent-steps|连续思维步数]]"),
    ("NeurIPS 2025", "[[neurips-paper|NeurIPS 2025]]"),
    ("NeurIPS 论文", "[[neurips-paper|NeurIPS 论文]]"),
    ("搜索前沿", "[[search-frontier|搜索前沿]]"),
    ("叠加态饱和", "[[saturation|叠加态饱和]]"),
    ("混合评估", "[[mixed-eval|混合评估]]"),
    ("推理跳数", "[[reasoning-hops|推理跳数]]"),
    ("并行 BFS", "[[parallel-bfs|并行 BFS]]"),
    ("并行搜楼", "[[parallel-bfs|并行搜楼]]"),
    ("分布匹配", "[[distribution-match|分布匹配]]"),
    ("图直径", "[[graph-diameter|图直径]]"),
    ("人造链", "[[synthetic-chain|人造链]]"),
    ("叠加态", "[[superposition|叠加态]]"),
    ("r = 0.543", "[[correlation|r = 0.543]]"),
    ("连续思维", "[[latent-steps|连续思维]]"),
    ("思考层数", "[[latent-steps|思考层数]]"),
    ("思考步数", "[[latent-steps|思考步数]]"),
    ("内部思考", "[[latent-steps|内部思考]]"),
    ("ProsQA", "[[prosqa|ProsQA]]"),
    ("Coconut", "[[coconut|Coconut]]"),
    ("000003", "[[latent-exp|000003]]"),
    ("准确率", "[[accuracy|准确率]]"),
    ("答对率", "[[accuracy|答对率]]"),
    ("边界", "[[boundary|边界]]"),
    ("饱和", "[[saturation|饱和]]"),
    ("latent 反馈系数", "[[latent-feedback|latent 反馈系数]]"),
    ("反馈系数", "[[latent-feedback|反馈系数]]"),
    ("acc@d", "[[acc-at-depth|acc@d]]"),
    ("步数=跳数", "[[acc-at-depth|步数=跳数]]"),
    ("只改模型数值", "[[latent-feedback|只改模型数值]]"),
    ("同质基线", "[[prosqa-extend|同质基线]]"),
]

_MARKUP_SPLIT = re.compile(r"(\[\[[^\]]+\]\])")


def apply_glossary_markup(text: str) -> str:
    if not text:
        return text
    parts = _MARKUP_SPLIT.split(text)
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        for phrase, repl in PHRASE_MARKUPS:
            if phrase in chunk:
                chunk = chunk.replace(phrase, repl)
        parts[i] = chunk
    return "".join(parts)


def markup_report(obj):
    if isinstance(obj, str):
        return apply_glossary_markup(obj)
    if isinstance(obj, list):
        return [markup_report(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(markup_report(item) for item in obj)
    if isinstance(obj, dict):
        return {key: markup_report(val) for key, val in obj.items()}
    return obj
