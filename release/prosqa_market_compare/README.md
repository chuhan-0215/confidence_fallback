# ProsQA 商用/开源模型对比实验包

给第二台开发机的运行说明（无需 GPU，纯 API 调用）。

## 0. 包内容

```
prosqa_market_compare/
├── prosqa_eval.py      # 主评测脚本（仅用 Python 标准库）
├── summarize.py        # 汇总 results/*.json → Markdown 报告
├── run_all.sh          # 一键跑全部实验
├── models.txt          # 模型清单（22 个，分 4 类，可增删）
├── .env                # API 密钥（私有，勿外传/勿提交 git）
├── data/
│   └── prosqa_test_graph_4_coconut.json   # ProsQA 测试集 419 题（tree-hop 4）
└── results/            # 评测结果输出目录
```

## 1. 环境要求

- Python ≥ 3.8（无第三方依赖，不需要 pip install）
- 能访问 `http://35.220.164.252:3888/v1`（公网中转接口）

## 2. 快速验证（2 分钟）

```bash
cd prosqa_market_compare
python3 prosqa_eval.py --models gpt-4.1-nano --n 3 --workers 3
```

能输出 `acc=... errors=0` 即环境 OK。

## 3. 正式运行

```bash
# 默认矩阵：22 个模型 × 100 题（seed=42 抽样），直答 + 旗舰 CoT 两轮
bash run_all.sh 100

# 全量 419 题（耗时约 1~2 小时，视网络与推理模型速度）
bash run_all.sh 419

# 只跑某一类/某几个模型
python3 prosqa_eval.py --models gpt-4.1 deepseek-reasoner qwen3-8b --n 100

# 只跑 CoT 提示（测"显式思维链"的 token 开销）
python3 prosqa_eval.py --prompt-mode cot --models gpt-4.1 claude-sonnet-4-5-20250929
```

断点续跑：同一模型+提示模式的结果文件若已存在，删掉对应
`results/prosqa_eval_<model>_<mode>.json` 即可重跑该模型；其他模型不受影响。

## 4. 生成对比报告

```bash
python3 summarize.py
# → results/对比报告.md（含 CF 本方法参考行：95.23% 准确率 / 平均补全 ≈14.2 tok）
```

## 5. 实验设计说明

| 类别 | 模型 | 目的 |
|---|---|---|
| A 推理增强型 | gpt-5-mini, deepseek-reasoner, qwq-32b, gemini-2.5-flash-thinking, claude-sonnet-4-5-thinking, kimi-k2-thinking, qwen3-30b-a3b-thinking | 证明"砸推理 token"路线：准确率提升的代价是 5~10 倍 token 开销与 30 倍延迟 |
| B 旗舰直答型 | gpt-4.1, gpt-4.1-mini, claude-sonnet-4-5, gemini-2.5-pro, grok-4-fast | 最强直答模型在多跳图推理上的天花板（预期 <50%） |
| C 开源/国产直答型 | deepseek-v3.2, qwen3-max, glm-4.6, kimi-k2.5, llama-3.3-70b, MiniMax-M2.5, mistral-large | 覆盖主流开源/国产旗舰，验证结论普适性 |
| D 轻量小模型 | qwen3-8b, llama-3.1-8b, phi-4, gpt-4.1-nano, qwen2.5-14b | 与 CF 的 0.5B 基座同量级对照：大 16~28 倍的模型仍然失败 |

对照基准（论文/本机复现）：
- **CF（本方法）**：95.23%（seed 99，τ=0.48；五种子均值 93.89%±0.76），平均补全 ≈14.2 tok，本地 0.5B 单卡 ~0.52s
- CoT 基线（论文 GPT-4o）：61.6%，平均 2,420 tok
- Coconut 基座（论文）：65.8%，平均 4,796 tok

评判方式：与 CF 流水线同一套规范化判分（句级归一化 + 精确/子串匹配），保证公平可比。

## 6. 注意事项

- 密钥仅授权本项目内部使用，打包传输后请勿上传公共仓库。
- 中转站偶发限流：脚本自带一次重试；个别模型大量失败可单独重跑。
- 推理型模型单题延迟 10~60s 属正常；可用 `--workers` 调并发。
- 更换评测集：替换 `data/` 下 JSON 即可（需含 `question`/`answer` 字段）。
