param(
    [string]$VenvPath = ".venv",
    [switch]$IncludeDeepLearning
)

$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        & py -3.12 -m venv $VenvPath
    } catch {
        try {
            & py -3.11 -m venv $VenvPath
        } catch {
            throw "Unable to create a virtual environment with py. Install Python 3.11 or 3.12 and retry."
        }
    }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m venv $VenvPath
} else {
    throw "Python launcher not found. Install Python 3.11 or 3.12 and retry."
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

$extras = if ($IncludeDeepLearning) { "[deep-learning]" } else { "" }
& $venvPython -m pip install -e ".$extras"
& $venvPython -m ipykernel install --user --name ml-curriculum --display-name "Python (ml-curriculum)"

Write-Host "Environment bootstrapped at $VenvPath"
