# Download confidence_fallback source code to Windows PC.
# Usage:
#   Set-ExecutionPolicy -Scope Process Bypass
#   & "$env:USERPROFILE\Downloads\download_confidence_fallback.ps1"

$ErrorActionPreference = "Stop"

# Use HTTP — server cert is not trusted by Windows for HTTPS on this IP.
$Url = "http://39.105.119.125/others/000002/confidence_fallback_github.zip"
$DownloadDir = Join-Path $env:USERPROFILE "Downloads"
$ZipPath = Join-Path $DownloadDir "confidence_fallback_github.zip"
$ExtractDir = Join-Path $DownloadDir "confidence_fallback"

Write-Host "Downloading confidence_fallback source..." -ForegroundColor Cyan
Write-Host "  URL: $Url"
Write-Host "  Save to: $ZipPath"
Write-Host ""

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

function Download-WithFallback {
    param([string]$Uri, [string]$Out)
    try {
        Invoke-WebRequest -Uri $Uri -OutFile $Out -UseBasicParsing
        return
    } catch {
        Write-Host "Invoke-WebRequest failed, trying curl.exe ..." -ForegroundColor Yellow
    }
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curl) {
        throw "Download failed. Install curl or use browser: $Uri"
    }
    & curl.exe -L -o $Out $Uri
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe download failed"
    }
}

Download-WithFallback -Uri $Url -Out $ZipPath

if (-not (Test-Path $ZipPath)) {
    throw "Zip not found after download: $ZipPath"
}

Write-Host "Extracting to: $ExtractDir" -ForegroundColor Cyan
if (Test-Path $ExtractDir) {
    Remove-Item -Recurse -Force $ExtractDir
}
Expand-Archive -Path $ZipPath -DestinationPath $DownloadDir -Force

$Nested = Join-Path $DownloadDir "release\confidence_fallback"
if (Test-Path $Nested) {
    Move-Item -Path $Nested -Destination $ExtractDir -Force
    Remove-Item -Recurse -Force (Join-Path $DownloadDir "release") -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host "  Zip:    $ZipPath"
Write-Host "  Folder: $ExtractDir"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd `"$ExtractDir`""
Write-Host "  python verify_install.py"
