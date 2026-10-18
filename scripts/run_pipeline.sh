#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

flake8 heartml tests
pytest
python -m heartml.data
python -m heartml.eda
python -m heartml.train "$@"
python -m heartml.predict sample_batch.json
