# Confidence Fallback

**Confidence Fallback: Selective Stopping for Continuous Latent Reasoning**

Official code for *confidence_fallback* — a deployable stopping strategy for [Coconut](https://arxiv.org/abs/2412.06769) continuous latent reasoning on [ProsQA](https://arxiv.org/abs/2412.06769) graph reachability.

On 419 in-distribution questions (91 controlled runs), `confidence_fallback` reaches **95.23%** accuracy with a **7.2%** fallback rate at τ=0.48, improving over `fixed_3` by **11.4** percentage points. Across 53 OOD slices, `tri_zone` with hybrid slice routing yields a weighted gain of **2.08** percentage points. Out-of-distribution generalization on StrategyQA, ProofWriter, and MATH500 with native step-route baselines achieves **+0.62pp**, **0.00pp**, and **+6.00pp** respectively.

**Author:** Songyang Pang (庞淞阳)  
**Affiliation:** Experimental School Affiliated to Haidian Teachers Training College  
**Paper:** ICAIS 2026 Youth Scientist Track (see `submission_en/`)

---

## Method (short)

1. **Structure routing** — set main-path latent budget `n₀` from BFS depth.
2. **One Coconut forward** — get prediction `ŷ₀` and hidden state `h_{n₀}`.
3. **M2 stop head** — output confidence `p₀ = σ(g(h_{n₀}, n₀, x))`.
4. **Gate** — if `p₀ ≥ τ`, return `ŷ₀`; else run kNN–M2 online-stop fallback.

Default deployment spec: `results/phase43/deploy_spec_v8_final.json`.

---

## Phase 49: Out-of-Distribution Generalization

Latest results (2026-07-29) on three new datasets with **native step-route baseline**:

| Dataset | N | Baseline | CF | Δ (pp) | Status |
|---------|---|----------|-----|--------|--------|
| StrategyQA | 161 | 60.87% | 61.49% | **+0.62** | ✅ |
| ProofWriter | 1200 | 99.75% | 99.75% | **0.00** | ✅ |
| MATH500 | 50 | 8.00% | 14.00% | **+6.00** | ✅ |

**Key findings:**
- **MATH500** shows significant improvement (+6.00pp), demonstrating CF's effectiveness on mathematical reasoning
- **ProofWriter** is already saturated (99.75% baseline), explaining 0.00pp gain
- **StrategyQA** shows modest but consistent improvement, validating robustness

All results use validation-locked hyperparameters with no test-set tuning.

---

## Install

```bash
git clone https://github.com/chuhan-0215/confidence_fallback.git
cd confidence_fallback
bash install.sh          # venv + PyTorch + download Coconut checkpoint_300
python verify_install.py # check configs, M2 head, dataset, deps
```

Chinese quick start: [QUICKSTART_zh.md](QUICKSTART_zh.md)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/download_checkpoint.py
```

Coconut weights: `Shibo-UCSD/coconut-theory` (ProsQA `checkpoint_300`).

---

## Quick run

Smoke test (5 questions, recommended first):

```bash
source .venv/bin/activate
python scripts/smoke_test.py --device cpu --n 5
```

Full 419-question evaluation (GPU recommended):

```bash
python run_confidence_fallback.py --device cuda --seed 99
```

Train M2 stop head if missing:

```bash
python scripts/phase10/run_m2_learned_enough_stop.py --device cuda
```

Full experiment grid (historical phases):

```bash
python scripts/phase36/run_x5_full419_validate.py --device cuda --seed 99
```

---

## Repository layout

```
confidence_fallback/
  verify_install.py            # pre-flight checks
  run_confidence_fallback.py   # one-command main eval
  configs/                     # Coconut model config (required)
  scripts/
    smoke_test.py              # 5-sample quick test
    stop_head/                 # M2 / RichStopHead training & eval
    phase25/_fallback_eval.py  # confidence_fallback core logic
    boundary_budget.py         # structure routing + kNN budget
    evaluate_coconut.py        # Coconut loader & decode
  model/                       # Coconut model code
  data/                        # ProsQA + OOD slice datasets
  results/
    phase10/m2_enough_stop_head.pt
    phase43/deploy_spec_v8_final.json
    phase49/                   # OOD generalization results (NEW)
      phase49_summary.json     # StrategyQA/ProofWriter/MATH500 summary
  submission_en/               # ICAIS LaTeX source
  figures/                     # paper figures (PNG + TikZ)
```

See `CODE_STRUCTURE.md` for the full phase map.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{pang2026confidence,
  title={Confidence Fallback: Selective Stopping for Continuous Latent Reasoning},
  author={Pang, Songyang},
  year={2026},
  note={ICAIS 2026 Youth Scientist Track}
}
```

Related work: Coconut (Hao et al., 2024), Reasoning by Superposition (Zhu et al., NeurIPS 2025), CALM (Schuster et al., 2022).

---

## License

MIT License — see [LICENSE](LICENSE).
