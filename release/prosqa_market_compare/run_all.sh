#!/usr/bin/env bash
# 一键运行全部对比实验。
# 用法: bash run_all.sh [题数]    默认 100
set -u
cd "$(dirname "$0")"
N="${1:-100}"

echo "== 第一轮：全部模型 · 直答提示（direct）=="
python3 prosqa_eval.py --n "$N" --prompt-mode direct

echo "== 第二轮：旗舰与推理模型 · CoT 提示（token 开销对比）=="
python3 prosqa_eval.py --n "$N" --prompt-mode cot \
  --models gpt-4.1 claude-sonnet-4-5-20250929 gemini-2.5-pro deepseek-v3.2 qwen3-max

echo "== 汇总报告 =="
python3 summarize.py
