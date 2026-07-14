#!/bin/bash
# 实验五完成后运行：更新规律 JSON 并刷新页面内嵌结论
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python3 scripts/pattern_analysis.py
.venv/bin/python3 scripts/build_findings_summary.py
echo "Done. Refresh /others/000002/ to see updated pattern tables."
