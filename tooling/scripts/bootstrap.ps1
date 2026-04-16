param(
    [string]$VenvPath = ".venv",
    [switch]$IncludeDeepLearning
)

$ErrorActionPreference = "Stop"

function Test-SupportedPythonVersion {
    param([string]$VersionText)

    return $VersionText -match "^Python 3\.(11|12)(\.|$)"
}

function New-VenvWithInterpreter {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    & $Command @Arguments -m venv $VenvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
        throw "Interpreter invocation failed: $Command $($Arguments -join ' ')"
    }
}

$venvCreated = $false
$errors = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($versionFlag in @("-3.12", "-3.11")) {
        try {
            New-VenvWithInterpreter -Command "py" -Arguments @($versionFlag)
            $venvCreated = $true
            break
        } catch {
            $errors += $_.Exception.Message
        }
    }

    if (-not $venvCreated) {
        $pyList = & py --list 2>$null
        $providerMatches = @(
            $pyList |
                Select-String -Pattern "-V:([^\s]+).+Python 3\.(11|12)\b" |
                ForEach-Object { $_.Matches[0].Groups[1].Value }
        )

        foreach ($provider in $providerMatches) {
            try {
                New-VenvWithInterpreter -Command "py" -Arguments @("-V:$provider")
                $venvCreated = $true
                break
            } catch {
                $errors += $_.Exception.Message
            }
        }
    }
}

if (-not $venvCreated -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $pythonVersion = & python --version 2>$null
    if (Test-SupportedPythonVersion -VersionText $pythonVersion) {
        New-VenvWithInterpreter -Command "python" -Arguments @()
        $venvCreated = $true
    }
}

if (-not $venvCreated) {
    throw "Unable to create a virtual environment with Python 3.11 or 3.12. Install a supported runtime and retry. Checked py launcher entries and python.exe. Errors: $($errors -join ' | ')"
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

$extras = if ($IncludeDeepLearning) { "[deep-learning]" } else { "" }
& $venvPython -m pip install -e ".$extras"
& $venvPython -m ipykernel install --user --name ml-curriculum --display-name "Python (ml-curriculum)"

Write-Host "Environment bootstrapped at $VenvPath"
