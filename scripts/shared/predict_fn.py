"""Standard predict_fn for stop_head / boundary_budget."""
from __future__ import annotations

from run_adaptive_stop_experiment import predict_at_n


def make_predict_fn(model, tokenizer, device, profile):
    def predict_fn(m, t, s, n, d, seed, eval_profile):
        return predict_at_n(m, t, s, n, d, seed=seed, eval_profile=eval_profile)

    return predict_fn
