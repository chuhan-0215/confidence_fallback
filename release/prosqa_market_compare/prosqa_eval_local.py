#!/usr/bin/env python3
"""ProsQA 小模型本地评测（与 API 版 prosqa_eval.py 同一协议）。

在带 GPU 的机器（如 A100）上运行，评测与 CF 基座同量级的开源小模型。
默认模型：Qwen2.5-0.5B / 1.5B / 3B Instruct（0.5B 即 CF 基座同款，最公平对照）。

用法:
  python3 prosqa_eval_local.py                      # 跑默认 3 个模型
  python3 prosqa_eval_local.py --models Qwen/Qwen2.5-0.5B-Instruct
  python3 prosqa_eval_local.py --data /path/prosqa_test_graph_4_coconut.json

依赖: pip install transformers torch accelerate
国内机器建议先: export HF_ENDPOINT=https://hf-mirror.com
结果写入 results_local/prosqa_local_<model>_direct.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_local"

DIRECT_SUFFIX = (
    "\n\nAnswer the final question with exactly one short sentence, "
    "e.g. 'Sally is a sterpus.' Do not explain."
)

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").strip().lower().rstrip("."))


def judge(item: dict, answer_text: str) -> bool:
    gold = norm(item["answer"])
    pred = norm(answer_text)
    if not pred:
        return False
    if pred == gold:
        return True
    return gold in pred


def eval_model(model_id: str, data: list[dict], max_new_tokens: int,
               batch_size: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows: list[dict] = []
    t_start = time.time()

    for beg in range(0, len(data), batch_size):
        batch = data[beg:beg + batch_size]
        prompts = []
        for item in batch:
            msgs = [{"role": "user", "content": item["question"] + DIRECT_SUFFIX}]
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        elapsed = time.time() - t0
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        prompt_tok = int(enc["input_ids"].ne(tokenizer.pad_token_id).sum())
        comp_tok = int(sum((g != tokenizer.pad_token_id).sum().item() for g in gen))
        per_q = elapsed / len(batch)
        for item, text in zip(batch, texts):
            rows.append({
                "idx": item["_i"], "ok": judge(item, text.strip()),
                "text": text.strip(),
                "prompt_tokens": prompt_tok // len(batch),
                "completion_tokens": comp_tok // len(batch),
                "latency_sec": round(per_q, 2),
            })
        done = min(beg + batch_size, len(data))
        print(f"  [{model_id}] {done}/{len(data)}", flush=True)

    n = len(rows)
    acc = sum(1 for r in rows if r["ok"]) / n if n else 0.0
    pt = [r["prompt_tokens"] for r in rows]
    ct = [r["completion_tokens"] for r in rows]
    lat = [r["latency_sec"] for r in rows]
    return {
        "model": model_id,
        "prompt_mode": "direct",
        "n": n,
        "accuracy": round(acc, 4),
        "avg_prompt_tokens": round(sum(pt) / n, 1),
        "avg_completion_tokens": round(sum(ct) / n, 1),
        "avg_total_tokens": round(sum(p + c for p, c in zip(pt, ct)) / n, 1),
        "avg_latency_sec": round(sum(lat) / n, 2),
        "elapsed_sec": round(time.time() - t_start, 1),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--data", default=str(ROOT / "data" / "prosqa_test_graph_4_coconut.json"))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    samples = [{**it, "_i": i} for i, it in enumerate(data)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"题数={len(samples)} 模型数={len(args.models)}", flush=True)
    for model_id in args.models:
        print(f"[model] {model_id}", flush=True)
        res = eval_model(model_id, samples, args.max_new_tokens, args.batch_size)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
        out = OUT_DIR / f"prosqa_local_{safe}_direct.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"  -> acc={res['accuracy']} total={res['avg_total_tokens']} tok "
              f"lat={res['avg_latency_sec']}s ({res['elapsed_sec']}s)", flush=True)
        del res
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
