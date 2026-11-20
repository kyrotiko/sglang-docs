#!/bin/bash
# Byte-level V4.1 host-page round-trip on A5.
# Sources the Ascend environment (the `acl` Python module lives there).

set -eo pipefail

SGLANG_ROOT=${SGLANG_ROOT:-$HOME/repos/sglang}
SGLANG_PATH=${SGLANG_PATH:-$SGLANG_ROOT/python}
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d "$SGLANG_PATH" ]]; then
  echo "ERROR: missing SGLANG_PATH: $SGLANG_PATH" >&2
  exit 1
fi

# Ascend set_env scripts are not nounset-safe; do not use set -u.
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export PYTHONPATH="$SGLANG_PATH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_HICACHE_HOST_MEM_BACKEND=${SGLANG_HICACHE_HOST_MEM_BACKEND:-acl}

echo "SGLANG_PATH=$SGLANG_PATH"
echo "HOST_MEM_BACKEND=$SGLANG_HICACHE_HOST_MEM_BACKEND"
"$PYTHON_BIN" -c 'import sglang; print("sglang:", sglang.__file__)'
"$PYTHON_BIN" "$SCRIPT_DIR/a5_hicache_page_roundtrip.py" "$@"
