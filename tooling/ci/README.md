# CI

Continuous integration runs on every push to `main` and every pull request via [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

## Current checks
- Python environment install from `pyproject.toml`
- `ruff` linting for Python files under `tooling/` and `modules/`
- `markdownlint-cli2` for Markdown files across the repository
- notebook smoke test via `python tooling/scripts/run_notebook_smoke_test.py --timeout 180`

## Notebook scope
The smoke test executes every notebook matching `modules/*/notebooks/*.ipynb` except explicit exclusions passed with `--exclude`.

This keeps the baseline broad enough to catch notebook drift while still allowing future GPU-heavy or long-running notebooks to be skipped narrowly in CI.

## Local reproduction
Run the same checks locally with:

```bash
python -m pip install -e .[dev]
python -m ruff check tooling modules
python tooling/scripts/run_notebook_smoke_test.py --timeout 180
npx markdownlint-cli2 "**/*.md"
```

If `npx` is unavailable, install `markdownlint-cli2` globally or use the GitHub Action as the source of truth for Markdown linting.
