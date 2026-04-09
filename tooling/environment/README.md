# Python Environment

This directory contains the reproducible Python setup for notebooks, labs, and lightweight validation.

## Files
- `../../pyproject.toml`: pinned Python dependencies for the core curriculum environment.
- `environment.yml`: optional Conda wrapper that installs the same project in editable mode.
- `../scripts/bootstrap.ps1`: Windows bootstrap script for a local virtual environment.
- `../scripts/bootstrap.sh`: macOS/Linux bootstrap script for a local virtual environment.
- `../scripts/run_notebook_smoke_test.py`: headless execution check for the baseline notebooks.

## Supported baseline
- Python `3.11` or `3.12`
- CPU-only environment by default
- Windows, macOS, Linux, and WSL

## Quick start with `pip`

### Windows PowerShell
```powershell
./tooling/scripts/bootstrap.ps1
.\.venv\Scripts\Activate.ps1
python tooling/scripts/run_notebook_smoke_test.py
```

To include PyTorch in the local environment:
```powershell
./tooling/scripts/bootstrap.ps1 -IncludeDeepLearning
```

### macOS / Linux / WSL
```bash
./tooling/scripts/bootstrap.sh
source .venv/bin/activate
python tooling/scripts/run_notebook_smoke_test.py
```

To include PyTorch in the local environment:
```bash
./tooling/scripts/bootstrap.sh .venv --deep-learning
```

## Manual installation
If you prefer to create the environment yourself:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m ipykernel install --user --name ml-curriculum --display-name "Python (ml-curriculum)"
```

For Windows, replace `source .venv/bin/activate` with `.\.venv\Scripts\Activate.ps1`.

## Conda option
Use Conda when you want the interpreter managed by Conda but the package set sourced from the repo:

```bash
conda env create -f tooling/environment/environment.yml
conda activate ml-curriculum
python tooling/scripts/run_notebook_smoke_test.py
```

## Dependency policy
- Core dependencies are pinned in `pyproject.toml` to reduce notebook drift.
- PyTorch is optional and installed through the `deep-learning` extra because GPU and CUDA requirements differ across machines.
- If a future module needs CUDA-specific wheels or accelerator-specific packages, document them separately instead of weakening the CPU baseline.

## Reproducibility check
The current smoke test executes these notebooks from a clean kernel:
- `modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb`
- `modules/01-optimization/notebooks/gradient-descent-basics.ipynb`
- `modules/02-statistical-learning/notebooks/bias-variance-simulation.ipynb`

Run them with:

```bash
python tooling/scripts/run_notebook_smoke_test.py
```

This is the minimum baseline for Card 1.3. Card 1.4 should move the same check into CI.
