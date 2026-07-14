"""评估与监督方式配置 — 控制 prompt 构造与答案匹配规则。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvalProfile:
    """单次评估使用的监督 / prompt 变体。"""

    # index: 预测节点索引（Coconut 默认）；symbol: 预测类型名；symbol_cap: 首字母大写
    answer_mode: str = "index"
    # coconut: 边列表 + [Q] target neg [R] root；fixed_edges: 不 shuffle 边序
    prompt_mode: str = "coconut"
    # random: target/neg 随机先后；target_first: target 在前
    choice_order: str = "random"
    max_new_tokens: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_mode": self.answer_mode,
            "prompt_mode": self.prompt_mode,
            "choice_order": self.choice_order,
            "max_new_tokens": self.max_new_tokens,
        }


DEFAULT_PROFILE = EvalProfile()


def parse_eval_profile(raw: Optional[Dict[str, Any]] = None) -> EvalProfile:
    if not raw:
        return DEFAULT_PROFILE
    return EvalProfile(
        answer_mode=str(raw.get("answer_mode", "index")),
        prompt_mode=str(raw.get("prompt_mode", "coconut")),
        choice_order=str(raw.get("choice_order", "random")),
        max_new_tokens=int(raw.get("max_new_tokens", 4)),
    )


def expected_answer(sample: dict, profile: EvalProfile) -> str:
    target = int(sample["target"])
    symbols = sample.get("idx_to_symbol") or []
    if profile.answer_mode == "symbol":
        name = symbols[target] if target < len(symbols) else str(target)
        return str(name).lower()
    if profile.answer_mode == "symbol_cap":
        name = symbols[target] if target < len(symbols) else str(target)
        return str(name)
    return str(target)


def profile_label(profile: EvalProfile) -> str:
    parts = []
    if profile.answer_mode != "index":
        parts.append(f"答案={profile.answer_mode}")
    if profile.prompt_mode != "coconut":
        parts.append(f"prompt={profile.prompt_mode}")
    if profile.choice_order != "random":
        parts.append(f"选项={profile.choice_order}")
    return " · ".join(parts) if parts else "标准索引监督"
