"""Load Coconut checkpoint and evaluate ProsQA graph reachability."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from coconut import Coconut  # noqa: E402
from stokenizer import STokenizer  # noqa: E402

from eval_profile import EvalProfile, expected_answer, parse_eval_profile  # noqa: E402
from graph_utils import build_eval_prompt, graph_diameter, reasoning_hops, root_to_target_distance  # noqa: E402


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def load_coconut_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    tokenizer = STokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(config_path)))
    model = Coconut(base, latent_id, start_id, end_id, tokenizer.eos_token_id)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    model.base_causallm.to(device)
    return model, tokenizer


def decode_answer(tokenizer: STokenizer, output_ids: torch.Tensor) -> str:
    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
    text = text.replace("<eos>", "").strip()
    answer_part = text.split("[A]")[-1].replace(",", "").strip()
    return answer_part.split()[0] if answer_part else ""


def evaluate_latent_steps(
    model,
    tokenizer: STokenizer,
    dataset: List[dict],
    n_latent: int,
    device: torch.device,
    seed: int = 0,
    sample_cb=None,
    eval_profile: EvalProfile | None = None,
) -> Dict:
    profile = eval_profile or parse_eval_profile(None)
    shuffle_edges = profile.prompt_mode != "fixed_edges"
    correct = 0
    total = 0
    forward_passes = 0
    latencies: List[float] = []
    dataset_len = len(dataset)

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            prompt = build_eval_prompt(
                sample,
                n_latent,
                seed=seed + idx,
                choice_order=profile.choice_order,
                shuffle_edges=shuffle_edges,
            )
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=device,
            )
            t0 = time.perf_counter()
            outputs = model.generate(
                input_ids,
                attention_mask=None,
                max_new_tokens=profile.max_new_tokens,
            )
            latencies.append(time.perf_counter() - t0)
            forward_passes += getattr(model, "gen_forward_cnt", 0)

            pred = decode_answer(tokenizer, outputs)
            if profile.answer_mode.startswith("symbol"):
                pred = pred.lower()
            expected = expected_answer(sample, profile)
            total += 1
            if pred == expected:
                correct += 1
            if sample_cb:
                sample_cb(total, dataset_len, n_latent)

    accuracy = correct / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    compute_cost = max(n_latent + 1, 1) * (forward_passes / max(total, 1))
    efficiency = accuracy / max(n_latent, 1)
    efficiency_compute = accuracy / max(compute_cost, 1e-6)

    row = {
        "n_latent": n_latent,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_latency_sec": round(avg_latency, 4),
        "avg_forward_passes": round(forward_passes / max(total, 1), 2),
        "efficiency": round(efficiency, 4),
        "efficiency_compute": round(efficiency_compute, 4),
    }
    if eval_profile is not None:
        row["eval_profile"] = profile.to_dict()
    return row


def load_dataset(path: Path, max_samples: Optional[int] = None) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if max_samples is not None:
        data = data[: max_samples]
    for i, row in enumerate(data):
        row["_idx"] = i
    return data


def dataset_meta(dataset: List[dict]) -> Dict:
    hops = [reasoning_hops(s) for s in dataset]
    dists = [root_to_target_distance(s) for s in dataset]
    diams = [graph_diameter(s) for s in dataset]
    return {
        "count": len(dataset),
        "mean_reasoning_hops": round(sum(hops) / len(hops), 3) if hops else 0,
        "mean_root_target_distance": round(sum(dists) / len(dists), 3) if dists else 0,
        "mean_diameter_from_root": round(sum(diams) / len(diams), 3) if diams else 0,
    }
