from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@torch.no_grad()
def extract_latent_hidden(
    model,
    input_ids: torch.Tensor,
    *,
    pass_idx: int,
) -> torch.Tensor:
    """Hidden vector after completing latent pass `pass_idx` (0-based)."""
    return _extract_latent_hidden_core(model, input_ids, pass_idx=pass_idx, detach=True)


def extract_latent_hidden_trainable(
    model,
    input_ids: torch.Tensor,
    *,
    pass_idx: int,
) -> torch.Tensor:
    """Differentiable variant for Coconut joint fine-tuning."""
    return _extract_latent_hidden_core(model, input_ids, pass_idx=pass_idx, detach=False)


def _extract_latent_hidden_core(
    model,
    input_ids: torch.Tensor,
    *,
    pass_idx: int,
    detach: bool,
) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).reshape(1, -1)
    labels = input_ids.clone()

    latent_indices = (input_ids == model.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max(len(lst) for lst in latent_lists)
    if pass_idx >= max_n_latents:
        raise ValueError(f"pass_idx {pass_idx} >= max_n_latents {max_n_latents}")

    next_compute_range = (0, input_ids.shape[1])
    inputs_embeds = model.embedding(input_ids)
    if max_n_latents > 0:
        next_compute_range = (0, latent_indices[:, 1].min().item())

    kv_cache = None
    hidden_states = None
    hidden_states_offset = 0

    for cur_pass in range(pass_idx + 1):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                output_hidden_states=True,
            )
            hidden_states_offset = 0
        else:
            past_key_values = [
                (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :])
                for k, v in kv_cache
            ]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask=attention_mask[:, : next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            hidden_states_offset = next_compute_range[0]

        next_compute_range = (
            next_compute_range[1],
            (
                input_ids.shape[1]
                if cur_pass + 1 >= max_n_latents
                else next_compute_range[1] + 1
            ),
        )
        hidden_states = outputs.hidden_states[-1]
        kv_cache = outputs.past_key_values

        filling_indices = [
            (instance_idx, mask_list[cur_pass])
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > cur_pass
        ]
        tensor_list = [
            [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
            for batch_idx in range(inputs_embeds.shape[0])
        ]
        for batch_idx, token_idx in filling_indices:
            old_embed = tensor_list[batch_idx][token_idx]
            hidden_fb = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
            tensor_list[batch_idx][token_idx] = model._mix_feedback(cur_pass, hidden_fb, old_embed)
        inputs_embeds = torch.stack(
            [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])]
        )

    batch_idx = 0
    token_idx = latent_lists[batch_idx][pass_idx]
    vec = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
    return vec.detach() if detach else vec

