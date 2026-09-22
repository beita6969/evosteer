#!/usr/bin/env bash
# Install the text-only ALFWorld environment and download its official data.
#
# usage: scripts/provision_alfworld.sh [python] [data_dir]
#   python    interpreter of the environment that runs EvoSteer (default: python)
#   data_dir  where ALFWorld data goes (default: $ALFWORLD_DATA or ~/.cache/alfworld)
#
# Every package already installed in that environment is pinned with a
# constraints file, so installing alfworld (which pulls textworld, jericho and
# spacy) cannot change the versions the text benchmarks were trained with.
# Recommended: run it on a copy of the training environment, as the paper runs
# did (venv313 -> venv313_alfworld).
set -euo pipefail
PY="${1:-python}"
DATA="${2:-${ALFWORLD_DATA:-$HOME/.cache/alfworld}}"
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT

"$PY" -m pip freeze | grep -v '^-e ' > "$CONSTRAINTS"
"$PY" -m pip install -c "$CONSTRAINTS" "alfworld==0.4.2"
"$PY" -c "import alfworld, textworld; print('alfworld OK, textworld', textworld.__version__)"

export ALFWORLD_DATA="$DATA"
mkdir -p "$ALFWORLD_DATA"
BIN="$(dirname "$("$PY" -c 'import sys; print(sys.executable)')")"
"$BIN/alfworld-download"
echo "ALFWorld data in $ALFWORLD_DATA"
echo "Before training: export ALFWORLD_DATA=$ALFWORLD_DATA"
