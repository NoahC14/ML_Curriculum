from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


DEFAULT_NOTEBOOKS = [
    Path("modules/00-math-toolkit/notebooks/linear-algebra-warmup.ipynb"),
    Path("modules/01-optimization/notebooks/gradient-descent-basics.ipynb"),
    Path("modules/02-statistical-learning/notebooks/bias-variance-simulation.ipynb"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute curriculum smoke-test notebooks.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-notebook execution timeout in seconds.")
    parser.add_argument("notebooks", nargs="*", help="Optional notebook paths to execute.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    notebooks = [Path(item) for item in args.notebooks] if args.notebooks else DEFAULT_NOTEBOOKS

    for notebook in notebooks:
        notebook_path = (root / notebook).resolve()
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        print(f"Executing {notebook}")
        execute_notebook(notebook_path, root, args.timeout)

    print(f"Executed {len(notebooks)} notebook(s) successfully.")
