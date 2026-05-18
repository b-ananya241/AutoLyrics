# AutoLyrics Phase 1 — create venv and install dependencies (Windows)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment ..."
    python -m venv .venv
}

$pip = Join-Path $Root ".venv\Scripts\pip.exe"
$python = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $pip install -U pip wheel

# CPU wheels by default; for CUDA 12.x replace with:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
Write-Host "Installing PyTorch (CPU) ..."
& $pip install torch torchaudio

Write-Host "Installing project requirements ..."
& $pip install -r requirements.txt

Write-Host "Running environment check ..."
& $python scripts\check_environment.py

Write-Host "`nDone. Activate with: .\.venv\Scripts\Activate.ps1"
