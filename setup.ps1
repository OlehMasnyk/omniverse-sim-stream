# Creates .venv and installs the NVIDIA runtimes. Run from the project root:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { py -3 -m venv .venv }
$python = Join-Path $PWD ".venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw "pip install failed" }

# Windows Smart App Control blocks Warp's unsigned CPU compiler (warp-clang.dll).
# Warp only loads it if present, and this project runs Warp on CUDA only.
$sac = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState
$clang = Join-Path $PWD ".venv\Lib\site-packages\warp\bin\warp-clang.dll"
if ($sac -eq 1 -and (Test-Path $clang)) {
    Rename-Item $clang "warp-clang.dll.blocked"
    Write-Host "Smart App Control is on: disabled warp-clang.dll (Warp CPU kernels unavailable)."
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
& $python -c "import ovrtx, ovstage, ovphysx, ovstream, warp; print('ovrtx', ovrtx.__version__, '| ovstage', ovstage.__version__, '| ovphysx', ovphysx.__version__, '| ovstream', ovstream.get_version())"
Write-Host "Setup complete. Start with: .\run.ps1"
