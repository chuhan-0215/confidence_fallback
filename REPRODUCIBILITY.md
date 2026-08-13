# Reproducibility Guide

This document lists everything needed to reproduce the main numbers in the ICAIS 2026 paper *Confidence Fallback: Selective Stopping for Continuous Latent Reasoning*.

Repository: https://github.com/chuhan-0215/confidence_fallback

---

## 1. Environment

```bash
git clone https://github.com/chuhan-0215/confidence_fallback.git
cd confidence_fallback
bash install.sh          # venv + PyTorch + Coconut checkpoint_300
python verify_install.py
```

- **Python:** 3.10+
- **Dependencies:** `requirements.txt` (torch 2.5.1, transformers 4.46.2, …)
- **Install script:** `install.sh` creates venv and downloads Coconut weights
- **Coconut weights:** `Shibo-UCSD/coconut-theory` — ProsQA `checkpoint_300`
- **Hardware:** GPU recommended for full 419-question eval (~1 h on a single A800); CPU works for `smoke_test.py`

---

## 2. Data and splits

| Item | Path |
|------|------|
| ProsQA test set (419 questions) | `data/prosqa_test_graph_4_coconut.json` |
| OOD / shift slices (53) | `data/` slice manifests (see `scripts/dataset_slice_specs.py`) |
| Stop-head train split | 60% of 419 (`split_dataset`, seed 42) |
| Stop-head val split | 20% of train (seed 43) |
| Canonical eval seed | **99** (locked in deploy spec) |

**Eval profile (all main numbers):**

```json
{
  "answer_mode": "index",
  "prompt_mode": "coconut",
  "choice_order": "random",
  "max_new_tokens": 4
}
```

---

## 3. Stop head (`RichStopHead`)

### Architecture

- Input: Coconut hidden state `h_n` (768-d), step index `n`, answer-bucket / streak / flip features
- Output: logit → `p = σ(g(h_n, n, s_n))`, where `s_n` = answer bucket, repeat streak, change flag
- Class: `scripts/stop_head/models.py` → `RichStopHead` (~106K trainable parameters)

### Training

```bash
python scripts/phase10/run_m2_learned_enough_stop.py --device cuda --epochs 40
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam, lr `1e-3`, weight decay `1e-4` |
| Loss | Focal BCE, γ = **2.0** |
| Pos weight | `neg / pos` (class imbalance; train: 201 pos / 1407 neg = 12.5% positive) |
| Epochs | 40 (early stop patience 10 on val loss) |
| Batch size | 64 |
| Dropout | 0.15 |
| Max latent steps (cap) | 8 |
| Label mode | `first_correct` / `is_correct` — stop when routed prediction first matches gold |
| Checkpoint | `results/phase10/m2_enough_stop_head.pt` |

### Threshold selection

- Sweep τ ∈ [0.42, 0.55] on validation split
- **τ = 0.48** chosen by balanced accuracy objective
- Five-seed τ sweep (seeds 42–46): mean accuracy **93.89%** (validation stability, not test tuning)

---

## 4. Confidence Fallback (in-distribution ProsQA)

### Policy

1. **Structure routing:** `n₀ = clamp(BFS_depth, n_min=2, n_max=8)`
2. **One Coconut forward** at `n₀` → prediction `ŷ₀`, hidden `h_{n₀}`
3. **Gate:** if `p₀ ≥ τ` return `ŷ₀`; else kNN–M2 online-stop fallback over neighbor depths

### Run

```bash
python run_confidence_fallback.py --device cuda --seed 99 --tau 0.48
```

### Paper numbers (seed 99, 419 questions)

| Metric | Value |
|--------|-------|
| Accuracy | **95.23%** (@ seed 99) |
| Five-seed mean ± std | **93.89% ± 0.76 pp** (seeds 0, 1, 2, 42, 99) |
| Δ vs fixed_3 (83.8%) | +11.4 pp |
| Refinement / fallback rate | **7.2%** |
| Mean latent steps `n̄` | **3.51** (fixed_3: 3.0; routing-only: 3.52) |
| Single-forward rate | 92.8% |

Deploy spec: `results/phase43/deploy_spec_v8_final.json`

---

## 5. Ablation ladder (Figure 2A)

Reproduce with phase-36 validation script:

```bash
python scripts/phase36/run_x5_full419_validate.py --device cuda --seed 99
```

| Policy | Acc (%) | Notes |
|--------|---------|-------|
| fixed_3 | 83.8 | uniform 3-step budget |
| auto_route (BFS) | 93.1 | frontier-based budget |
| structure_d (depth-aware) | 93.6 | shortest-path depth routing |
| knn_min3 (universal refine) | 92.6 | always search neighbors — **below routing** |
| confidence_fallback | **95.23** | route-then-gate |

**Order ablation:** gate-before-route and refine-every-instance variants plateau at 93.5–94.0% in the phase-35/36 grid, below CF (95.23%).

**Depth–structure correlation:** `r = 0.543` over 21 graph subsets → `results/pattern_laws.json`

---

## 6. Error decomposition

| Audit | Count | Source |
|-------|-------|--------|
| Routing-wrong, refinement-recoverable | 14 | complement audit |
| Routing-correct, refinement would mislead | 5 | complement audit |
| Both wrong | 17 | complement audit |
| Oracle union ceiling | 95.94% | +0.71 pp above CF |
| Structure-routing failures (fixed budget) | 22 | `results/phase24/y3_error_taxonomy_latest.json` |
| Commitment-path failures (fixed budget) | 31 | same file |

---

## 7. Distribution shift (53 slices)

Tri-band gating: `t_low = 0.40`, `t_mid = 0.48`  
Hybrid slice router: `hybrid_slice_router_v4` (default) / v5 (canonical @ seed 99)

| Metric | Value |
|--------|-------|
| Weighted Δ (53 slices) | **+2.08 pp** |
| In-distribution subset | +0.19 pp |
| OOD subset | **+7.44 pp** |
| Regressing slices | 6 / 53 |

Source: `results/phase38/deploy_spec_v4.json`, rollup in `results/phase43/deploy_spec_v8_final.json`

Reproduce OOD grid:

```bash
python scripts/phase38/run_z4_deploy_spec_v4_validate.py --device cuda
```

---

## 8. Cross-dataset transfer (Table 1 / Figure 2D)

Validation-locked τ and gating; **no test-time threshold search**.

| Setting | N | Baseline | CF | Δ (pp) |
|---------|---|----------|-----|--------|
| ProsQA | 419 | 83.8% | 95.23% | +11.4 |
| PrOntoQA | 800 | 99.50% | 99.62% | +0.12 |
| GSM8K (self) | 1319 | 27.07% | 28.35% | +1.28 |
| GSM8K cp14 | 1319 | 31.31% | 31.84% | +0.53 |
| GSM8K cp25 | 1319 | 32.98% | 33.66% | +0.68 |

- **PrOntoQA:** SC unanimity as confidence signal (same gate skeleton)
- **GSM8K:** when stop-head AUC < 0.5, plug in calibrated correctness classifier without retraining Coconut

Cross-transfer scripts: `scripts/findings/cross_transfer.py`, phase-44 OOD audit.

---

## 9. Commercial LLM comparison (Table 1)

Reference-only direct-answer API runs on ProsQA; **not** a controlled same-scale baseline.

| Setting | Value |
|---------|-------|
| Script | `release/prosqa_market_compare/prosqa_eval.py` |
| Prompt mode | `direct` (no CoT) |
| Random seed | **42** (subset sampling) |
| Subset size | 100 questions for Qwen3.7-Max, GPT-5.4, Kimi-K2.6 |
| Full split | 419 questions for GPT-4.1, GPT-4.1-Mini |
| Latency | End-to-end wall-clock per API call (includes network) |
| Token counts | API-reported prompt + completion tokens |

Paper Table 1 models (seed 42 where N=100):

| Model | N | Acc (%) | Avg tokens | Latency (s) |
|-------|---|---------|------------|-------------|
| CF (Qwen2.5-0.5B) | 419 | 95.2 | 14.2 | 0.52 |
| Qwen3.7-Max | 100 | 94.0 | 1163.9 | 26.06 |
| GPT-5.4 | 100 | 87.0 | 480.9 | 2.48 |
| Kimi-K2.6 | 100 | 77.0 | 484.8 | 4.53 |
| GPT-4.1 | 419 | 65.6 | 355.0 | 2.30 |
| GPT-4.1-Mini | 419 | 53.5 | 354.2 | 1.74 |

Raw JSON: `release/prosqa_market_compare/results/prosqa_eval_*_direct.json`

---

## 10. Deployment constraints

All reported policies satisfy:

- No per-question sweep over depths 1–8 at inference
- Mean latent steps **≤ 4.5**
- Hyperparameters locked on validation before test / slice eval

---

## 10. Key result files

```
results/phase10/m2_enough_stop_head.pt
results/phase43/deploy_spec_v8_final.json
results/pattern_laws.json
results/phase24/y3_error_taxonomy_latest.json
results/findings_summary.json
```

---

## Citation

```bibtex
@inproceedings{pang2026confidence,
  title={Confidence Fallback: Selective Stopping for Continuous Latent Reasoning},
  author={Pang, Songyang},
  year={2026},
  note={ICAIS 2026 Youth Scientist Track}
}
```
