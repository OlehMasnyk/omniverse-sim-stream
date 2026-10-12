# Starts the simulation server. Extra arguments are passed through, e.g.:
#   .\run.ps1 --width 1920 --height 1080
#   .\run.ps1 --public-ip 192.168.1.50      (view from another machine on the LAN)
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
& .\.venv\Scripts\python.exe -u -m server.main @args 2>&1 | Tee-Object -FilePath logs\server.log
