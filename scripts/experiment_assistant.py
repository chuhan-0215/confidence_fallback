"""Coconut 边界实验页 AI 助手 — 服务端调用 OpenAI 兼容中转 API。"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "results" / "findings_summary.json"
PATTERN_LAWS = ROOT / "results" / "pattern_laws.json"
ENV_CANDIDATES = [
    ROOT / "config" / "ai.env",
    ROOT.parents[1] / "three-kingdoms" / "tools" / ".env",
]


def _load_dotenv() -> None:
    for path in ENV_CANDIDATES:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2)
        break


def ai_config() -> dict:
    _load_dotenv()
    base = (
        os.environ.get("COCONUT_AI_BASE_URL")
        or os.environ.get("TK_AI_BASE_URL")
        or "http://35.220.164.252:3888/v1"
    ).rstrip("/")
    return {
        "base_url": base,
        "api_key": os.environ.get("COCONUT_AI_API_KEY") or os.environ.get("TK_AI_API_KEY") or "",
        "model": os.environ.get("COCONUT_AI_MODEL")
        or os.environ.get("TK_AI_PROMPT_MODEL")
        or "gpt-5-mini",
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_experiment_context() -> str:
    findings = _read_json(FINDINGS)
    laws = _read_json(PATTERN_LAWS)
    kn = findings.get("key_numbers") or {}
    parts: List[str] = []

    if findings.get("headline"):
        parts.append(f"总结论：{findings['headline']}")
    if findings.get("one_liner"):
        parts.append(f"一句话：{findings['one_liner']}")

    if kn:
        parts.append(
            "关键数字："
            f"全量边界 {kn.get('full_boundary_steps')} 步，"
            f"峰值准确率 {float(kn.get('full_peak_accuracy') or 0) * 100:.1f}%，"
            f"3步 {float(kn.get('acc_at_3_steps') or 0) * 100:.1f}%，"
            f"4步 {float(kn.get('acc_at_4_steps') or 0) * 100:.1f}%，"
            f"5步 {float(kn.get('acc_at_5_steps') or 0) * 100:.1f}%。"
        )

    why = findings.get("why_analysis") or {}
    if why.get("summary"):
        parts.append(f"归因摘要：{why['summary']}")

    mech = findings.get("mechanism_analysis") or {}
    if mech.get("lead"):
        parts.append(f"机制分析：{mech['lead']}")
    if mech.get("tldr"):
        parts.append(f"一句话：{mech['tldr']}")
    elif mech.get("conclusion"):
        parts.append(f"机制结论：{mech['conclusion']}")

    for law in (laws.get("laws") or findings.get("pattern_laws", {}).get("laws") or [])[:6]:
        parts.append(
            f"- {law.get('title')}：{law.get('pattern')} 原因：{law.get('why')}"
        )

    mix = laws.get("mix_ladder") or {}
    if mix.get("summary"):
        parts.append(f"混合比例实验：{mix['summary']}")

    cross = laws.get("hop_diameter_cross") or {}
    if cross.get("summary"):
        parts.append(f"跳数×直径：{cross['summary']}")

    if laws.get("unified_conclusion"):
        parts.append(f"统一因果链：{laws['unified_conclusion']}")

    extreme = findings.get("extreme_cases") or {}
    if extreme.get("tldr"):
        parts.append(f"极端准确率：{extreme['tldr']}")
    for row in (extreme.get("rows") or [])[:8]:
        parts.append(
            f"- {row.get('label')}（{row.get('supervision')}）：峰值 {row.get('max_accuracy')} · {row.get('category')}"
        )

    for exp in (findings.get("experiments") or [])[:5]:
        parts.append(
            f"实验 {exp.get('title')}：边界 {exp.get('boundary')}，{exp.get('note')}"
        )

    return "\n".join(parts)[:12000]


def build_system_prompt() -> str:
    ctx = build_experiment_context()
    return f"""你是「Coconut 连续思维边界实验」页面（/others/000002/）的专属 AI 助手。

你的职责：基于下方实验数据与 Reasoning by Superposition（NeurIPS 2025）论文机制，用清晰中文回答用户关于本实验的问题。

回答要求：
1. 优先引用页面上的实验结论、六条规律与具体数字；不确定时说明「当前数据未覆盖」。
2. 解释「为什么」时，区分：论文机制（BFS/叠加态）、数据深度、混合比例、训练分布、加步饱和。
3. 边界不是固定常数 3——在 ProsQA 混合集上常报 3 步，纯 4 跳子集报 4 步，真实延长 5 跳可达 5 步。
4. 不要编造未在数据中出现的准确率；可概括趋势但不要虚构子集结果。
5. 若问题与 Coconut/ProsQA/连续思维边界无关，简短说明并引导回实验主题。
6. 子集最高准确率为 0 或接近 0 时，先区分：符号监督（格式不匹配）还是 OOD 人工图（分布不匹配）；此类边界勿解读。
7. 回答简洁有条理，可用短段落或少量列表，避免过长。
7. 直接输出面向用户的中文结论，不要输出空回复或仅内部推理过程。

=== 实验数据摘要（自动生成）===
{ctx}
=== 数据摘要结束 ==="""


def assistant_status() -> dict:
    cfg = ai_config()
    return {
        "ok": bool(cfg["api_key"]),
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "has_context": FINDINGS.is_file(),
    }


def _extract_reply(data: dict) -> tuple[str, str]:
    """从 chat/completions 响应中提取可见回复。返回 (text, finish_reason)。"""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = str(choice.get("finish_reason") or "")

    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    text = str(content or "").strip()

    if not text:
        text = str(message.get("reasoning_content") or "").strip()

    return text, finish


def _call_chat(cfg: dict, messages: List[dict], max_tokens: int) -> dict:
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.55,
        "max_completion_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_with_assistant(user_messages: List[dict]) -> dict:
    cfg = ai_config()
    if not cfg["api_key"]:
        return {"ok": False, "error": "AI API 密钥未配置（检查 three-kingdoms/tools/.env）"}

    cleaned: List[dict] = []
    for msg in user_messages or []:
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            cleaned.append({"role": role, "content": content[:4000]})
    if not cleaned or cleaned[-1]["role"] != "user":
        return {"ok": False, "error": "需要至少一条用户消息"}

    cleaned = cleaned[-24:]
    messages = [{"role": "system", "content": build_system_prompt()}] + cleaned

    token_limits = [3200, 4800]
    last_finish = ""
    last_usage = None
    for attempt, max_tokens in enumerate(token_limits):
        try:
            data = _call_chat(cfg, messages, max_tokens)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body)
                msg = err.get("error", {}).get("message") or err.get("message") or body[:300]
            except json.JSONDecodeError:
                msg = body[:300] or f"HTTP {exc.code}"
            return {"ok": False, "error": msg}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        reply, last_finish = _extract_reply(data)
        last_usage = data.get("usage")
        if reply:
            return {
                "ok": True,
                "reply": reply,
                "model": cfg["model"],
                "usage": last_usage,
            }

        # 推理模型偶发占满 token 导致 content 为空；追加简短指令重试
        if attempt + 1 < len(token_limits):
            messages = messages + [
                {
                    "role": "user",
                    "content": "请直接给出简洁的中文回答，不要留空。",
                }
            ]

    hint = "（推理 token 用尽）" if last_finish == "length" else ""
    return {
        "ok": False,
        "error": f"模型返回空内容{hint}，请稍后重试或缩短问题",
    }
