#!/usr/bin/env bash
# 重跑失败模型（新平台等价模型）
cd "$(dirname "$0")"
N="${1:-100}"

echo "[$(date '+%H:%M:%S')] == 开始重跑失败模型 =="

# Direct 模式：10 个新平台等价模型
echo "[$(date '+%H:%M:%S')] == Direct 模式（10个模型）=="
python3 prosqa_eval.py --n "$N" --prompt-mode direct \
  --models deepseek-v4-pro qwen3.7-max glm-5.2 kimi-k2.6 MiniMax-M3 qwen3.7-plus MiniMax-M2.7 gpt-5.4 kimi-k3

# CoT 模式：5 个新平台等价模型
echo "[$(date '+%H:%M:%S')] == CoT 模式（5个模型）=="
python3 prosqa_eval.py --n "$N" --prompt-mode cot \
  --models claude-sonnet-5 deepseek-v4-pro gemini-3.5-flash gpt-5.4 qwen3.7-max

echo "[$(date '+%H:%M:%S')] == 全部完成，生成汇总 =="
python3 summarize.py
echo "[$(date '+%H:%M:%S')] == 结束 =="
