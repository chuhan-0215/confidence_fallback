# Confidence Fallback

Research code for the ICAIS 2026 paper *Confidence Fallback: Selective Stopping for Continuous Latent Reasoning* (Songyang Pang).

The repository implements route-then-gate stopping for frozen [Coconut](https://arxiv.org/abs/2412.06769) models on ProsQA graph reachability. On 419 in-distribution questions, the default policy reaches **95.23%** accuracy with a **7.2%** refinement rate at validation-locked $\tau=0.48$ (+11.4 percentage points over a fixed three-step budget). On 53 shifted ProsQA slices, tri-band gating yields a weighted gain of **+2.08** percentage points.

Paper source: `submission_en/icais_bundle/`  
Reproducibility notes: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## Setup

```bash
git clone https://github.com/chuhan-0215/confidence_fallback.git
cd confidence_fallback
bash install.sh
python verify_install.py
```

`install.sh` creates a virtual environment, installs PyTorch and dependencies, and downloads Coconut `checkpoint_300` from Hugging Face (`Shibo-UCSD/coconut-theory`). Chinese notes: [QUICKSTART_zh.md](QUICKSTART_zh.md).

## Reproduce main numbers

Smoke test (5 questions):

```bash
source .venv/bin/activate
python scripts/smoke_test.py --device cpu --n 5
```

Full ProsQA evaluation (GPU recommended):

```bash
python run_confidence_fallback.py --device cuda --seed 99
```

Train the stop head if the checkpoint is missing:

```bash
python scripts/phase10/run_m2_learned_enough_stop.py --device cuda
```

The locked deployment settings live in `results/phase43/deploy_spec_v8_final.json`.

## Layout

```
confidence_fallback/
  run_confidence_fallback.py   # main ProsQA eval entry point
  verify_install.py
  scripts/stop_head/           # RichStopHead training code
  scripts/phase25/             # confidence-fallback inference core
  model/                       # Coconut implementation
  data/                        # ProsQA and slice manifests
  results/phase43/             # deploy spec and checkpoints
  submission_en/               # ICAIS LaTeX source
```

See `CODE_STRUCTURE.md` for the phase history.

## Citation

```bibtex
@inproceedings{pang2026confidence,
  title={Confidence Fallback: Selective Stopping for Continuous Latent Reasoning},
  author={Pang, Songyang},
  year={2026},
  note={ICAIS 2026 Youth Scientist Track}
}
```

## License

MIT — see [LICENSE](LICENSE).
