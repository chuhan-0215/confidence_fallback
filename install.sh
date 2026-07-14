#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -U pip -q
  .venv/bin/pip install numpy==2.1.3 transformers==4.46.2 huggingface_hub tqdm \
    -i https://pypi.tuna.tsinghua.edu.cn/simple -q
fi

if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  mkdir -p wheels
  WHEEL=wheels/torch-2.5.1+cpu-cp312-cp312-linux_x86_64.whl
  if [[ ! -f "$WHEEL" ]]; then
    echo "Downloading PyTorch CPU wheel..."
    curl -L --retry 5 -o "$WHEEL" \
      "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp312-cp312-linux_x86_64.whl"
  fi
  .venv/bin/pip install --no-deps "$WHEEL"
fi

if [[ ! -f checkpoints/checkpoint_300 ]]; then
  .venv/bin/python scripts/download_checkpoint.py
fi

echo "Environment ready."
