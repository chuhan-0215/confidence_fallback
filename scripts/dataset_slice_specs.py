"""Dataset slice specification tables (data-only)."""
from __future__ import annotations

from typing import List

_MIX_SEED = 2026
_MIX_CAP = 80

SLICE_SPECS: List[dict] = [
    {
        "id": "full",
        "label": "全量 ProsQA",
        "description": "419 条 Coconut 测试集，3/4 跳各约一半",
        "filter_name": None,
        "default_cap": 100,
    },
    {
        "id": "hops_3",
        "label": "仅 3 跳推理链",
        "description": "ground-truth 推理链长度为 3 的样本",
        "filter_name": "hops_3",
        "default_cap": None,
    },
    {
        "id": "hops_4",
        "label": "仅 4 跳推理链",
        "description": "ground-truth 推理链长度为 4 的样本",
        "filter_name": "hops_4",
        "default_cap": None,
    },
    {
        "id": "diameter_3",
        "label": "图直径 = 3",
        "description": "从根出发图直径为 3 的子图",
        "filter_name": "diameter_3",
        "default_cap": 80,
    },
    {
        "id": "diameter_4",
        "label": "图直径 = 4",
        "description": "从根出发图直径为 4 的子图",
        "filter_name": "diameter_4",
        "default_cap": 80,
    },
    {
        "id": "diameter_wide",
        "label": "图直径 ≥ 5",
        "description": "更宽图结构（直径 5–6），样本较少",
        "filter_name": "diameter_wide",
        "default_cap": None,
    },
    {
        "id": "random_a",
        "label": "随机子集 A",
        "description": "seed=42 随机抽取，检验采样波动",
        "filter_name": "random_a",
        "default_cap": 80,
    },
    {
        "id": "random_b",
        "label": "随机子集 B",
        "description": "seed=7 随机抽取，与 A 对照",
        "filter_name": "random_b",
        "default_cap": 80,
    },
    {
        "id": "first_half",
        "label": "前半段",
        "description": "原 JSON 前 210 条（文件顺序）",
        "filter_name": "first_half",
        "default_cap": None,
    },
    {
        "id": "second_half",
        "label": "后半段",
        "description": "原 JSON 后 209 条（文件顺序）",
        "filter_name": "second_half",
        "default_cap": None,
    },
]

# 深边界实验：5–6 跳合成图 + ProsQA 宽图对照
DEEP_SLICE_SPECS: List[dict] = [
    {
        "id": "syn_chain_5",
        "label": "合成纯 5 跳链",
        "description": "人工构造 5 层推理链，根→目标 BFS=5",
        "source": "data/synthetic_chain_5hop.json",
        "default_cap": None,
    },
    {
        "id": "syn_chain_6",
        "label": "合成纯 6 跳链",
        "description": "人工构造 6 层推理链，根→目标 BFS=6",
        "source": "data/synthetic_chain_6hop.json",
        "default_cap": None,
    },
    {
        "id": "syn_chain_5_wide",
        "label": "合成 5 跳 + 干扰分支",
        "description": "5 跳主链 + 侧枝死胡同，检验宽图是否抬高边界",
        "source": "data/synthetic_chain_5hop_wide.json",
        "default_cap": None,
    },
    {
        "id": "syn_chain_6_wide",
        "label": "合成 6 跳 + 干扰分支",
        "description": "6 跳主链 + 侧枝，图更宽",
        "source": "data/synthetic_chain_6hop_wide.json",
        "default_cap": None,
    },
    {
        "id": "syn_mixed_56",
        "label": "合成 5/6 跳混合",
        "description": "30 条 5 跳 + 30 条 6 跳",
        "source": "data/synthetic_mixed_56hop.json",
        "default_cap": None,
    },
    {
        "id": "prosqa_diameter_5",
        "label": "ProsQA 图直径=5",
        "description": "原测试集中直径 5 的样本（约 19 条）",
        "filter_name": "diameter_5",
        "default_cap": None,
    },
    {
        "id": "prosqa_diameter_6",
        "label": "ProsQA 图直径=6",
        "description": "原测试集中直径 6 的样本（约 3 条）",
        "filter_name": "diameter_6",
        "default_cap": None,
    },
    {
        "id": "prosqa_diameter_wide",
        "label": "ProsQA 直径≥5（对照）",
        "description": "原宽图子集，上次实验边界为 4 步",
        "filter_name": "diameter_wide",
        "default_cap": None,
    },
]

DEEP_SLICE_IDS = {s["id"] for s in DEEP_SLICE_SPECS}

# 构造 × 监督对照：多种图结构 + 评估方式变体
VARIANT_SLICE_SPECS: List[dict] = [
    {
        "id": "v_real_hops_3",
        "label": "真实 3 跳（基线）",
        "description": "ProsQA 原集 3 跳，标准索引监督",
        "filter_name": "hops_3",
        "default_cap": 50,
        "construction": "real",
        "eval_profile": {},
    },
    {
        "id": "v_real_hops_4",
        "label": "真实 4 跳（基线）",
        "description": "ProsQA 原集 4 跳，标准索引监督",
        "filter_name": "hops_4",
        "default_cap": 50,
        "construction": "real",
        "eval_profile": {},
    },
    {
        "id": "v_chain_3",
        "label": "纯链 3 跳",
        "description": "人工纯链，无侧枝",
        "source": "data/variants/chain_3hop.json",
        "construction": "pure_chain",
        "eval_profile": {},
    },
    {
        "id": "v_chain_4",
        "label": "纯链 4 跳",
        "description": "人工纯链 4 层",
        "source": "data/variants/chain_4hop.json",
        "construction": "pure_chain",
        "eval_profile": {},
    },
    {
        "id": "v_chain_5_dense",
        "label": "稠密链 5 跳",
        "description": "5 跳主链 + 借自真实图的噪声边",
        "source": "data/variants/chain_5hop_dense.json",
        "construction": "dense_chain",
        "eval_profile": {},
    },
    {
        "id": "v_chain_6_dense",
        "label": "稠密链 6 跳",
        "description": "6 跳主链 + 噪声边",
        "source": "data/variants/chain_6hop_dense.json",
        "construction": "dense_chain",
        "eval_profile": {},
    },
    {
        "id": "v_chain_7_dense",
        "label": "稠密链 7 跳",
        "description": "7 跳主链 + 噪声边",
        "source": "data/variants/chain_7hop_dense.json",
        "construction": "dense_chain",
        "eval_profile": {},
    },
    {
        "id": "v_extend_5",
        "label": "真实图延长 → 5 跳",
        "description": "在 ProsQA 样本末尾追加推理层",
        "source": "data/variants/prosqa_extend_5hop.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "v_extend_6",
        "label": "真实图延长 → 6 跳",
        "description": "保留原图结构，链长 6",
        "source": "data/variants/prosqa_extend_6hop.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "v_extend_7",
        "label": "真实图延长 → 7 跳",
        "description": "保留原图结构，链长 7",
        "source": "data/variants/prosqa_extend_7hop.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "v_tree_5",
        "label": "树形 5 跳",
        "description": "左链到目标 + 右分支干扰",
        "source": "data/variants/tree_5hop.json",
        "construction": "tree",
        "eval_profile": {},
    },
    {
        "id": "v_diamond_5",
        "label": "菱形 5 跳",
        "description": "双路径汇合到目标",
        "source": "data/variants/diamond_5hop.json",
        "construction": "diamond",
        "eval_profile": {},
    },
    {
        "id": "v_chain_6_symbol",
        "label": "稠密 6 跳 · 符号监督",
        "description": "同一数据，期望答案改为类型名而非索引",
        "source": "data/variants/chain_6hop_dense.json",
        "construction": "dense_chain",
        "eval_profile": {"answer_mode": "symbol", "max_new_tokens": 8},
    },
    {
        "id": "v_chain_6_fixed",
        "label": "稠密 6 跳 · 固定边序",
        "description": "prompt 中边列表不 shuffle",
        "source": "data/variants/chain_6hop_dense.json",
        "construction": "dense_chain",
        "eval_profile": {"prompt_mode": "fixed_edges"},
    },
    {
        "id": "v_extend_6_symbol",
        "label": "延长 6 跳 · 符号监督",
        "description": "真实延长图 + 符号名答案",
        "source": "data/variants/prosqa_extend_6hop.json",
        "construction": "prosqa_extend",
        "eval_profile": {"answer_mode": "symbol", "max_new_tokens": 8},
    },
    {
        "id": "v_real_4_symbol",
        "label": "真实 4 跳 · 符号监督",
        "description": "原集 4 跳 + 符号名答案",
        "filter_name": "hops_4",
        "default_cap": 50,
        "construction": "real",
        "eval_profile": {"answer_mode": "symbol", "max_new_tokens": 8},
    },
]

VARIANT_SLICE_IDS = {s["id"] for s in VARIANT_SLICE_SPECS}

# 第六轮：边界上推 — 同质基线跳数延长到 5/6 跳
BOUNDARY_PUSH_SLICE_SPECS: List[dict] = [
    {
        "id": "push_ext5_from4",
        "label": "4跳基线 → 延长5跳",
        "description": "仅从原 4 跳样本末尾 +1 层，同质 5 跳",
        "source": "data/variants/prosqa_extend_5hop_from4.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext6_from4",
        "label": "4跳基线 → 延长6跳",
        "description": "仅从原 4 跳样本末尾 +2 层，同质 6 跳",
        "source": "data/variants/prosqa_extend_6hop_from4.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext5_from3",
        "label": "3跳基线 → 延长5跳",
        "description": "仅从原 3 跳样本末尾 +2 层，同质 5 跳",
        "source": "data/variants/prosqa_extend_5hop_from3.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext6_from3",
        "label": "3跳基线 → 延长6跳",
        "description": "仅从原 3 跳样本末尾 +3 层，同质 6 跳",
        "source": "data/variants/prosqa_extend_6hop_from3.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext5_mixed",
        "label": "混合基线 → 5跳（对照）",
        "description": "3/4 跳混合延长，与第四轮 v_extend_5 相同数据",
        "source": "data/variants/prosqa_extend_5hop.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext6_mixed",
        "label": "混合基线 → 6跳（对照）",
        "description": "3/4 跳混合延长，与第四轮 v_extend_6 相同数据",
        "source": "data/variants/prosqa_extend_6hop.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_mix_56_from4",
        "label": "4跳延长 5/6 跳混合",
        "description": "30 条 5 跳 + 30 条 6 跳，均从 4 跳基线延长",
        "source": "data/variants/prosqa_mix_56_from4.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
]

BOUNDARY_PUSH_DEEP_SLICE_SPECS: List[dict] = [
    {
        "id": "push_ext5_from3",
        "label": "3跳基线 → 5跳（对照）",
        "description": "第六轮已验证：边界 5、acc@5 高",
        "source": "data/variants/prosqa_extend_5hop_from3.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext6_from3",
        "label": "3跳基线 → 6跳（对照）",
        "description": "第六轮：能力可迁移，报边界易偏低",
        "source": "data/variants/prosqa_extend_6hop_from3.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext7_from3",
        "label": "3跳基线 → 7跳",
        "description": "同质延长 +4 层，检验边界能否到 7 且 acc@7 不跌",
        "source": "data/variants/prosqa_extend_7hop_from3.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext8_from3",
        "label": "3跳基线 → 8跳",
        "description": "同质延长 +5 层，检验 8 步深度",
        "source": "data/variants/prosqa_extend_8hop_from3.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
    {
        "id": "push_ext7_mixed",
        "label": "混合基线 → 7跳（对照）",
        "description": "3/4 跳混合延长到 7 跳",
        "source": "data/variants/prosqa_extend_7hop_mixed.json",
        "construction": "prosqa_extend",
        "eval_profile": {},
    },
]

BOUNDARY_PUSH_SLICE_IDS = {s["id"] for s in BOUNDARY_PUSH_SLICE_SPECS}
BOUNDARY_PUSH_DEEP_SLICE_IDS = {s["id"] for s in BOUNDARY_PUSH_DEEP_SLICE_SPECS}

# 第五轮：混合比例阶梯 + 跳数×直径交叉（寻规律）
PATTERN_SLICE_SPECS: List[dict] = [
    {
        "id": "mix_0_4",
        "label": "混合 0% 4跳（纯3跳）",
        "description": "80 条全为 3 跳，检验纯浅集边界",
        "filter_name": "mix_0_4",
        "default_cap": _MIX_CAP,
        "pattern_axis": "mix_ratio",
        "ratio_4hop": 0.0,
    },
    {
        "id": "mix_25_4",
        "label": "混合 25% 4跳",
        "description": "60 条 3 跳 + 20 条 4 跳",
        "filter_name": "mix_25_4",
        "default_cap": _MIX_CAP,
        "pattern_axis": "mix_ratio",
        "ratio_4hop": 0.25,
    },
    {
        "id": "mix_50_4",
        "label": "混合 50% 4跳",
        "description": "40 条 3 跳 + 40 条 4 跳，接近全量分布",
        "filter_name": "mix_50_4",
        "default_cap": _MIX_CAP,
        "pattern_axis": "mix_ratio",
        "ratio_4hop": 0.5,
    },
    {
        "id": "mix_75_4",
        "label": "混合 75% 4跳",
        "description": "20 条 3 跳 + 60 条 4 跳",
        "filter_name": "mix_75_4",
        "default_cap": _MIX_CAP,
        "pattern_axis": "mix_ratio",
        "ratio_4hop": 0.75,
    },
    {
        "id": "mix_100_4",
        "label": "混合 100% 4跳（纯4跳）",
        "description": "80 条全为 4 跳",
        "filter_name": "mix_100_4",
        "default_cap": _MIX_CAP,
        "pattern_axis": "mix_ratio",
        "ratio_4hop": 1.0,
    },
    {
        "id": "hop3_diam3",
        "label": "3跳 · 直径3",
        "description": "推理链 3 跳且图直径 3（155 条可用）",
        "filter_name": "hop3_diam3",
        "default_cap": 60,
        "pattern_axis": "hop_diameter",
    },
    {
        "id": "hop3_diam4",
        "label": "3跳 · 直径4",
        "description": "推理链 3 跳但图更宽（直径 4，41 条）",
        "filter_name": "hop3_diam4",
        "default_cap": None,
        "pattern_axis": "hop_diameter",
    },
    {
        "id": "hop4_diam4",
        "label": "4跳 · 直径4",
        "description": "推理链 4 跳且图直径 4（201 条）",
        "filter_name": "hop4_diam4",
        "default_cap": 60,
        "pattern_axis": "hop_diameter",
    },
    {
        "id": "hop4_diam5",
        "label": "4跳 · 直径≥5",
        "description": "推理链 4 跳但图更宽（直径 5–6，16 条）",
        "filter_name": "hop4_diam5",
        "default_cap": None,
        "pattern_axis": "hop_diameter",
    },
]

PATTERN_SLICE_IDS = {s["id"] for s in PATTERN_SLICE_SPECS}

# Phase 45：AlienBench_v1 — 刻意与 ProsQA 不同的拓扑/命名/跳数
ALIEN_SLICE_SPECS: List[dict] = [
    {
        "id": "alien_full",
        "label": "AlienBench 全量",
        "description": "80 条陌生题：星型/梯子/沙漏/宽树，模块命名",
        "source": "data/alien/alien_benchmark.json",
        "construction": "alien",
        "default_cap": None,
    },
    {
        "id": "alien_star",
        "label": "Alien · 星型枢纽",
        "description": "枢纽 + 长链，BFS 深度与 steps 可不一致",
        "source": "data/alien/alien_star.json",
        "construction": "alien_star",
        "default_cap": None,
    },
    {
        "id": "alien_ladder",
        "label": "Alien · 双轨梯子",
        "description": "双轨并行 + 横档，非链式 ProsQA 拓扑",
        "source": "data/alien/alien_ladder.json",
        "construction": "alien_ladder",
        "default_cap": None,
    },
    {
        "id": "alien_hourglass",
        "label": "Alien · 沙漏",
        "description": "分叉→汇合→再分叉，菱形变体",
        "source": "data/alien/alien_hourglass.json",
        "construction": "alien_hourglass",
        "default_cap": None,
    },
    {
        "id": "alien_tree",
        "label": "Alien · 宽树",
        "description": "每层二分叉宽树，深度 3–5",
        "source": "data/alien/alien_tree.json",
        "construction": "alien_tree",
        "default_cap": None,
    },
]

ALIEN_SLICE_IDS = {s["id"] for s in ALIEN_SLICE_SPECS}

