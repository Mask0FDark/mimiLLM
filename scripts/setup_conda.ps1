$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda was not found. Install Miniconda or Anaconda and retry."
}

$exists = conda env list --json | ConvertFrom-Json
$hasEnvironment = $exists.envs | Where-Object { (Split-Path $_ -Leaf) -eq "mimillm" }
if ($hasEnvironment) {
    Write-Host "Updating the mimillm environment from environment-windows.yml..."
    conda env update -n mimillm -f environment-windows.yml
} else {
    Write-Host "Creating the mimillm Windows environment..."
    conda env create -f environment-windows.yml
}

Write-Host "Installing mimiLLM in editable mode..."
conda run -n mimillm python -m pip install --no-deps --no-build-isolation -e .

Write-Host "Ready. Run these commands in PowerShell:"
Write-Host "  conda activate mimillm"
Write-Host "  python tools/build_backend.py --release"
Write-Host "  python -m unittest discover -s tests -v"
