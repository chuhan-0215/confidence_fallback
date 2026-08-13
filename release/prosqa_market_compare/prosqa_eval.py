#!/usr/bin/env python3
"""ProsQA 商用/开源模型对比评测（多模型 · 可复现）。

用法:
  python3 prosqa_eval.py                     # 跑 models.txt 全部模型
  python3 prosqa_eval.py --models gpt-4.1 deepseek-reasoner
  python3 prosqa_eval.py --n 419             # 全量
  python3 prosqa_eval.py --prompt-mode cot   # 显式思维链提示（对比 token 开销）
  python3 prosqa_eval.py --list              # 列出可选模型预设

只依赖 Python 标准库。结果写入 results/prosqa_eval_<model>_<mode>.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "prosqa_test_graph_4_coconut.json"
OUT_DIR = ROOT / "results"
MODELS_FILE = ROOT / "models.txt"
ENV_FILE = ROOT / ".env"

DIRECT_SUFFIX = (
    "\n\nAnswer the final question with exactly one short sentence, "
    "e.g. 'Sally is a sterpus.' Do not explain."
)
COT_SUFFIX = (
    "\n\nThink step by step, then give the final answer on the last line "
    "as exactly one short sentence, e.g. 'Sally is a sterpus.'"
)


def load_env() -> tuple[str, str]:
    base, key = "", ""
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
            if not m:
                continue
            if m.group(1) == "TK_AI_BASE_URL":
                base = m.group(2).strip()
            elif m.group(1) == "TK_AI_API_KEY":
                key = m.group(2).strip()
    return (base.rstrip("/") or "http://35.220.164.252:3888/v1"), key


def call_model(base: str, key: str, model: str, question: str, prompt_mode: str) -> dict:
    suffix = COT_SUFFIX if prompt_mode == "cot" else DIRECT_SUFFIX
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question + suffix}],
    }
    payload = None
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            req = urlreq.Request(
                f"{base}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"},
            )
            t0 = time.time()
            with urlreq.urlopen(req, timeout=300) as resp:
                payload = json.load(resp)
            latency = time.time() - t0
            break
        except HTTPError as e:
            detail = e.read().decode(errors="ignore")[:300]
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            # 部分推理模型拒绝默认参数，第二次尝试去掉所有可选字段重试
            if attempt == 1:
                time.sleep(2)
                continue
            raise last_err
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == 1:
                time.sleep(2)
                continue
            raise
    msg = payload["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text and isinstance(msg.get("reasoning_content"), str):
        text = msg["reasoning_content"].strip()[-500:]
    usage = payload.get("usage") or {}
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_sec": round(latency, 2),
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().rstrip("."))


def judge(item: dict, answer_text: str) -> bool:
    gold = norm(item["answer"])
    pred = norm(answer_text)
    if not pred:
        return False
    if pred == gold:
        return True
    return gold in pred


def eval_model(base: str, key: str, model: str, samples: list[dict],
               prompt_mode: str, workers: int) -> dict:
    rows: list[dict] = []

    def one(item: dict) -> dict:
        try:
            r = call_model(base, key, model, item["question"], prompt_mode)
            return {"idx": item["_i"], "ok": judge(item, r["text"]), **r}
        except Exception as e:  # noqa: BLE001
            return {"idx": item["_i"], "ok": False, "error": str(e)[:200],
                    "text": "", "prompt_tokens": None, "completion_tokens": None,
                    "latency_sec": None}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, it) for it in samples]
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"  [{model}] {done}/{len(samples)}", flush=True)

    valid = [r for r in rows if r.get("prompt_tokens") is not None]
    acc = sum(1 for r in rows if r["ok"]) / len(rows) if rows else 0.0
    pt = [r["prompt_tokens"] for r in valid]
    ct = [r["completion_tokens"] for r in valid]
    lat = [r["latency_sec"] for r in valid]
    return {
        "model": model,
        "prompt_mode": prompt_mode,
        "n": len(rows),
        "n_ok_api": len(valid),
        "accuracy": round(acc, 4),
        "avg_prompt_tokens": round(sum(pt) / len(pt), 1) if pt else None,
        "avg_completion_tokens": round(sum(ct) / len(ct), 1) if ct else None,
        "avg_total_tokens": round(sum(p + c for p, c in zip(pt, ct)) / len(pt), 1) if pt else None,
        "avg_latency_sec": round(sum(lat) / len(lat), 2) if lat else None,
        "errors": len(rows) - len(valid),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="模型 id 列表；缺省读 models.txt")
    ap.add_argument("--n", type=int, default=100, help="抽样题数（≤419）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-mode", choices=["direct", "cot"], default="direct")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list", action="store_true", help="打印 models.txt 并退出")
    args = ap.parse_args()

    if args.list:
        print(MODELS_FILE.read_text())
        return

    base, key = load_env()
    if not key:
        print("未找到 API key：请编辑包内 .env（TK_AI_API_KEY=...）", file=sys.stderr)
        sys.exit(1)

    if args.models:
        models = args.models
    else:
        models = [ln.strip() for ln in MODELS_FILE.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]

    data = json.loads(DATA.read_text())
    n = min(args.n, len(data))
    idxs = random.Random(args.seed).sample(range(len(data)), n)
    samples = [{**data[i], "_i": i} for i in idxs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"题数={n} seed={args.seed} 提示模式={args.prompt_mode} 模型数={len(models)}", flush=True)
    for model in models:
        print(f"[model] {model}", flush=True)
        t0 = time.time()
        try:
            res = eval_model(base, key, model, samples, args.prompt_mode, args.workers)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {model} 整体失败: {e}", flush=True)
            continue
        res["elapsed_sec"] = round(time.time() - t0, 1)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        out = OUT_DIR / f"prosqa_eval_{safe}_{args.prompt_mode}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"  -> acc={res['accuracy']} completion={res['avg_completion_tokens']} tok "
              f"errors={res['errors']} ({res['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
