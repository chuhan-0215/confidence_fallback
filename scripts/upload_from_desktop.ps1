# ICAIS 投稿文件上传脚本（在 Windows PowerShell 中运行）
# 用法：右键「使用 PowerShell 运行」，或在 PowerShell 中执行：
#   Set-ExecutionPolicy -Scope Process Bypass; & "$env:USERPROFILE\...\upload_from_desktop.ps1"
# 也可复制到桌面后双击运行（需已配置 ssh/scp）

$ErrorActionPreference = "Stop"

$Server = "admin@39.105.119.125"
$RemoteDir = "/home/admin/chu-han/others/000002/inbox/upload"
$Desktop = [Environment]::GetFolderPath("Desktop")

# 待上传文件（按你桌面实际文件名自动匹配）
$Candidates = @(
    @{ Local = "icais.zip"; Remote = "icais.zip"; Required = $true },
    @{ Local = "庞淞阳-ICAIS2026_Track2_少年科学家投稿.docx"; Remote = "庞淞阳-ICAIS2026_Track2_少年科学家投稿.docx"; Required = $true },
    @{ Local = "庞淞阳-ICAIS2026_Track2_少年科学家投稿.pdf"; Remote = "庞淞阳-ICAIS2026_Track2_少年科学家投稿.pdf"; Required = $false },
    @{ Local = "庞淞阳-ICAIS2026_Track2_少年科学家投稿"; Remote = "庞淞阳-ICAIS2026_Track2_少年科学家投稿.docx"; Required = $false }
)

Write-Host "桌面路径: $Desktop"
Write-Host "目标服务器: ${Server}:${RemoteDir}"
Write-Host ""

# 确保远端目录存在
ssh $Server "mkdir -p $RemoteDir"

$Uploaded = 0
foreach ($item in $Candidates) {
    $path = Join-Path $Desktop $item.Local
    if (-not (Test-Path $path)) { continue }

    # 若「无扩展名」条目实际是 .docx，则按 docx 上传
    $remoteName = $item.Remote
    if ($item.Local -eq "庞淞阳-ICAIS2026_Track2_少年科学家投稿" -and $path -notlike "*.docx") {
        if (Test-Path ($path + ".docx")) { $path = $path + ".docx" }
    }

    Write-Host "上传: $path"
    scp "`"$path`"" "${Server}:${RemoteDir}/${remoteName}"
    if ($LASTEXITCODE -ne 0) { throw "scp 失败: $path" }
    $Uploaded++
}

if ($Uploaded -eq 0) {
    Write-Host ""
    Write-Host "未找到可上传文件。请确认桌面存在：" -ForegroundColor Yellow
    Write-Host "  - icais.zip"
    Write-Host "  - 庞淞阳-ICAIS2026_Track2_少年科学家投稿.docx"
    exit 1
}

Write-Host ""
Write-Host "上传完成（$Uploaded 个文件）。" -ForegroundColor Green
Write-Host "请在 Cursor 中告诉助手：「已上传，请处理 inbox/upload」"
Write-Host ""
Write-Host "也可手动触发服务器处理："
Write-Host "  ssh $Server 'bash /home/admin/chu-han/others/000002/scripts/process_submission_inbox.sh'"
