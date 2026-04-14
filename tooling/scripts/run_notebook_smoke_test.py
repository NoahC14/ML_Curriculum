from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DEFAULT_NOTEBOOK_GLOB = "modules/*/notebooks/*.ipynb"
DEFAULT_EXCLUDE_PATTERNS = [
    "*/.ipynb_checkpoints/*",
]


def execute_notebook(path: Path, root: Path, timeout: int) -> None:
    with path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str((root / path.parent).resolve())}},
    )
    client.execute()


def discover_notebooks(root: Path, exclude_patterns: list[str]) -> list[Path]:
    notebooks = []
    for path in sorted(root.glob(DEFAULT_NOTEBOOK_GLOB)):
        relative_path = path.relative_to(root)
        relative_path_text = relative_path.as_posix()
        if any(fnmatch.fnmatch(relative_path_text, pattern) for pattern in exclude_patterns):
            continue
        notebooks.append(relative_path)
    return notebooks


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute curriculum smoke-test notebooks.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-notebook execution timeout in seconds.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern for notebooks to skip. Repeat the flag to add multiple patterns.",
    )
    parser.add_argument("notebooks", nargs="*", help="Optional notebook paths to execute.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    exclude_patterns = [*DEFAULT_EXCLUDE_PATTERNS, *args.exclude]
    notebooks = (
        [Path(item) for item in args.notebooks]
        if args.notebooks
        else discover_notebooks(root, exclude_patterns)
    )

    if not notebooks:
        raise FileNotFoundError("No notebooks matched the smoke-test selection.")

    for notebook in notebooks:
        notebook_path = (root / notebook).resolve()
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        print(f"Executing {notebook}")
        execute_notebook(notebook_path, root, args.timeout)

    print(f"Executed {len(notebooks)} notebook(s) successfully.")
