#!/usr/bin/env bash
# 处理 inbox/upload：解压官方模板、同步用户 Word、生成英文稿
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPLOAD="$ROOT/inbox/upload"
TEMPLATE_DIR="$ROOT/submission_template"
EN_DIR="$ROOT/submission_en"

mkdir -p "$UPLOAD" "$TEMPLATE_DIR" "$EN_DIR"

echo "==> 1. 解压官方模板 icais.zip"
ZIP=""
for z in "$UPLOAD/icais.zip" "$UPLOAD/CAIS 2026 TEMPLATE.zip" "$ROOT/inbox/icais.zip"; do
  if [[ -f "$z" ]]; then ZIP="$z"; break; fi
done
if [[ -n "$ZIP" ]]; then
  rm -rf "$TEMPLATE_DIR"/*
  python3 - "$ZIP" "$TEMPLATE_DIR" <<'PY'
import sys
import zipfile
from pathlib import Path
zpath = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zpath) as z:
    z.extractall(out)
print(f"extracted {zpath} -> {out}")
PY
  find "$TEMPLATE_DIR" -maxdepth 3 -type f \( -name '*.tex' -o -name '*.sty' \) | head -15
else
  echo "    未找到 icais.zip，跳过（请用 upload_from_desktop.ps1 上传）"
fi

echo "==> 2. 同步用户微调 Word"
USER_DOCX=""
for f in \
  "$UPLOAD/庞淞阳-ICAIS2026_Track2_少年科学家投稿.docx" \
  "$ROOT/inbox/庞淞阳-ICAIS2026_Track2_少年科学家投稿.docx"; do
  if [[ -f "$f" ]]; then USER_DOCX="$f"; break; fi
done
if [[ -n "$USER_DOCX" ]]; then
  python3 "$ROOT/scripts/sync_user_docx.py" "$USER_DOCX"
else
  echo "    未找到用户 docx，保留当前 ICAIS2026_Track2_少年科学家投稿.txt"
fi

echo "==> 3. 生成中文版 Word（若 build 脚本存在）"
if [[ -f "$ROOT/scripts/build_submission_docx.py" ]]; then
  python3 "$ROOT/scripts/build_submission_docx.py" || true
fi

echo "==> 4. 生成英文 LaTeX 稿"
python3 "$ROOT/scripts/build_submission_en.py"
python3 "$ROOT/scripts/build_submission_icais.py"

echo "==> 完成"
echo "    中文 txt : $ROOT/ICAIS2026_Track2_少年科学家投稿.txt"
echo "    英文 tex : $EN_DIR/paper.tex"
echo "    ICAIS 包 : $EN_DIR/icais_bundle/icais_submission.tex"
echo "    Overleaf zip: $EN_DIR/ICAIS2026_English_Overleaf.zip"
if [[ -f "$EN_DIR/paper.pdf" ]]; then
  echo "    英文 pdf : $EN_DIR/paper.pdf"
fi
