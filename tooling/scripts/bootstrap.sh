#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-.venv}"
EXTRAS=""

if [[ "${2:-}" == "--deep-learning" ]]; then
  EXTRAS="[deep-learning]"
fi

python3.12 -m venv "$VENV_PATH" 2>/dev/null || python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -e ".$EXTRAS"
"$VENV_PATH/bin/python" -m ipykernel install --user --name ml-curriculum --display-name "Python (ml-curriculum)"

echo "Environment bootstrapped at $VENV_PATH"
