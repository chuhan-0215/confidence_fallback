#!/usr/bin/env python3
"""汇总 results/ 下所有评测结果，生成 Markdown 对比报告。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

CATEGORY = {
    # 旧平台已完成模型
    "gpt-5-mini": "A", "deepseek-reasoner": "A", "qwq-32b": "A",
    "gemini-2.5-flash-thinking": "A", "claude-sonnet-4-5-20250929-thinking": "A",
    "kimi-k2-thinking": "A", "qwen3-30b-a3b-thinking-2507": "A",
    "gpt-4.1": "B", "gpt-4.1-mini": "B", "claude-sonnet-4-5-20250929": "B",
    "gemini-2.5-pro": "B", "grok-4-fast": "B",
    "deepseek-v3.2": "C", "qwen3-max": "C", "glm-4.6": "C", "kimi-k2.5": "C",
    "llama-3.3-70b-instruct": "C", "MiniMax-M2.5": "C", "mistral-large-2411": "C",
    "qwen3-8b": "D", "llama-3.1-8b-instruct": "D", "microsoft/phi-4": "D",
    "gpt-4.1-nano": "D", "qwen2.5-14b-instruct": "D",
    # 新平台重跑模型（等价映射）
    "deepseek-v4-pro": "C", "qwen3.7-max": "C", "glm-5.2": "C", "kimi-k2.6": "C",
    "MiniMax-M3": "C", "qwen3.7-plus": "C",
    "MiniMax-M2.7": "D", "gpt-5.4": "D", "kimi-k3": "D",
    # CoT 模式新平台模型
    "claude-sonnet-5": "B", "gemini-3.5-flash": "B",
}
CAT_NAME = {"A": "推理增强型", "B": "旗舰直答型", "C": "开源/国产直答型", "D": "轻量小模型", "?": "其他"}

CF_REFERENCE = {
    "method": "CF（置信度回退，本方法）",
    "base": "Qwen2.5-0.5B 微调基座",
    "accuracy": 0.9523, "avg_total_tokens": 14.2, "latency_note": "≈0.52s（本地单卡）",
    "source": "Phase 25 冠军（seed 99, τ=0.48, 419 题全量；五种子均值 93.89%±0.76）",
}


def main() -> None:
    rows = []
    for p in sorted(RES.glob("prosqa_eval_*.json")):
        d = json.loads(p.read_text())
        if d.get("n_ok_api", 0) == 0:
            continue
        rows.append(d)
    rows.sort(key=lambda r: (CATEGORY.get(r["model"], "?"),
                             -(r["accuracy"] or 0)))

    lines = [
        "# ProsQA 图推理：CF 与市售/开源模型对比（自动生成）",
        "",
        f"- 评测集：`prosqa_test_graph_4_coconut`（按 seed 抽样，详见各 JSON 的 n）",
        f"- CF 参考值：{CF_REFERENCE['source']}",
        "",
        "| 类别 | 模型 | 提示 | 准确率 | 平均提示 tok | 平均补全 tok | 平均总 tok | 平均延迟(s) | 调用失败 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| 本方法 | {CF_REFERENCE['method']}（{CF_REFERENCE['base']}） | — | "
        f"**{CF_REFERENCE['accuracy']*100:.1f}%** | — | — | **{CF_REFERENCE['avg_total_tokens']}** | "
        f"{CF_REFERENCE['latency_note']} | 0 |",
    ]
    for r in rows:
        cat = CAT_NAME.get(CATEGORY.get(r["model"], "?"), "其他")
        lines.append(
            f"| {cat} | {r['model']} | {r.get('prompt_mode','direct')} | "
            f"{(r['accuracy'] or 0)*100:.1f}% | {r['avg_prompt_tokens']} | "
            f"{r['avg_completion_tokens']} | {r['avg_total_tokens']} | "
            f"{r['avg_latency_sec']} | {r['errors']} |"
        )
    out = RES / "对比报告.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
