# Linting

Store formatting and linting configuration here.

## Active checks
- `ruff check tooling modules` for Python import ordering and lint errors
- `markdownlint-cli2 "**/*.md"` for Markdown structure and formatting
- notebook execution is handled separately by `../scripts/run_notebook_smoke_test.py`

## Configuration sources
- `../../pyproject.toml` contains the `ruff` configuration and baseline Python dependencies.
- `../../.markdownlint-cli2.jsonc` contains the repository Markdown lint rules.

## Notes
- Markdown line-length enforcement is disabled because the curriculum includes tables, code fences, and citation-heavy prose.
- If a future directory needs a lint exemption, prefer a narrow pattern in the relevant config instead of weakening checks repo-wide.
