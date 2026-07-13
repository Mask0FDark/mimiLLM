$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda was not found. Install Miniconda or Anaconda and retry."
}

$exists = conda env list --json | ConvertFrom-Json
$hasEnvironment = $exists.envs | Where-Object { (Split-Path $_ -Leaf) -eq "minillm" }
if ($hasEnvironment) {
    Write-Host "Updating the minillm environment from environment-windows.yml..."
    conda env update -n minillm -f environment-windows.yml
} else {
    Write-Host "Creating the minillm Windows environment..."
    conda env create -f environment-windows.yml
}

Write-Host "Ready. Run these commands in PowerShell:"
Write-Host "  conda activate minillm"
Write-Host "  python tools/build_backend.py --release"
Write-Host "  python -m unittest discover -s tests -v"
