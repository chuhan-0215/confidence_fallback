#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

bash install.sh
exec .venv/bin/python experiment_server.py
