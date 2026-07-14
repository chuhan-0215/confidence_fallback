# 快速上手（中文）

## 1. 下载并解压

PowerShell：

```powershell
# 用 HTTP（Windows 不信任该 IP 的 HTTPS 证书）
$Url = "http://39.105.119.125/others/000002/confidence_fallback_github.zip"
$Dir = "$env:USERPROFILE\Downloads"
$Zip = Join-Path $Dir "confidence_fallback_github.zip"
Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
Expand-Archive -Path $Zip -DestinationPath $Dir -Force
Move-Item "$Dir\release\confidence_fallback" "$Dir\confidence_fallback" -Force
Remove-Item "$Dir\release" -Recurse -Force -ErrorAction SilentlyContinue
cd "$Dir\confidence_fallback"
```

## 2. 安装环境

**Windows：**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/download_checkpoint.py
python verify_install.py
```

**Linux / macOS：**

```bash
bash install.sh
python verify_install.py
```

## 3. 跑测试

**冒烟测试（5 题，约几分钟，推荐先跑）：**

```bash
python scripts/smoke_test.py --device cpu --n 5
```

**完整 419 题主实验（GPU 推荐）：**

```bash
python run_confidence_fallback.py --device cuda --seed 99
```

## 包里有什么 / 缺什么

| 项目 | 状态 |
|------|------|
| 实验代码 `scripts/` | 已包含 |
| Coconut 模型代码 `model/` | 已包含 |
| 模型配置 `configs/` | **已包含** |
| ProsQA 数据 `data/` | 已包含 |
| M2 停步头权重 | 已包含 |
| 部署配置 `deploy_spec_v8_final.json` | 已包含 |
| Coconut 主权重 `checkpoint_300`（约 60MB） | **需下载**（见上） |

## 常见问题

**Q: `verify_install.py` 报 checkpoint 缺失？**  
A: 正常。先运行 `python scripts/download_checkpoint.py`。

**Q: CPU 太慢？**  
A: 先用 `smoke_test.py --n 5`；全量请用 `--device cuda`。

**Q: 如何上传 GitHub？**  
A: 在解压目录执行 `git init && git add . && git commit -m "Initial release"` 后 push。
