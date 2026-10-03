#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${ALFWORLD_PYTHON:-$(command -v python3)}"
DATA_ROOT="${ALFWORLD_DATA:-data/alfworld}"
CONFIG_FILE="${ALFWORLD_CONFIG_FILE:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ALFWorld Python not found: $PYTHON_BIN" >&2
  exit 2
fi
mkdir -p "$DATA_ROOT"
"$PYTHON_BIN" -m pip install 'alfworld[full]'
export ALFWORLD_DATA="$DATA_ROOT"
"$PYTHON_BIN" -m alfworld.agents.modules.generic --help >/dev/null 2>&1 || true
if command -v alfworld-download >/dev/null 2>&1; then
  alfworld-download
else
  "$PYTHON_BIN" -m alfworld.scripts.alfworld_download
fi

if [[ -n "$CONFIG_FILE" ]]; then
  export ALFWORLD_CONFIG_FILE="$CONFIG_FILE"
fi
"$PYTHON_BIN" - <<'PY'
import glob, os
root = os.environ["ALFWORLD_DATA"]
games = sorted(glob.glob(os.path.join(root, "json_2.1.1", "train", "*", "*", "game.tw-pddl")))
print({"data_root": root, "training_game_count": len(games), "first_game": games[0] if games else None})
if not games:
    raise SystemExit("ALFWorld download completed without training game files")
PY
