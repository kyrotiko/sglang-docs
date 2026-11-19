#!/bin/bash
# Target-only DeepSeek-V4.1 HiCache L2 E2E on A5.
#
# DSpark is intentionally NOT enabled: the draft SWA uses the V4 608 B row,
# while the V4.1 target uses 544 B, so both cannot share one SWA host pool.
# Use this script first; add a draft SWA sidecar only after this passes.
#
# Required:
#   MODEL_PATH      model directory (default below)
#   SGLANG_ROOT     source checkout containing python/sglang
#   HICACHE_RATIO   > 1.0, so device eviction can be forced while A stays in host
#   FILL_FACTOR     > 1.0 and < HICACHE_RATIO
#
# Example:
#   HICACHE_RATIO=1.2 FILL_FACTOR=1.05 \
#   SGLANG_ROOT=/mnt/share/y00627667/repos/sglang \
#   ./run_a5_hicache_target_e2e.sh

set -eo pipefail

MODEL_PATH=${MODEL_PATH:-/mnt/share/weight/DeepSeek-V4.1-Flash-1017}
SGLANG_ROOT=${SGLANG_ROOT:-$HOME/repos/sglang}
SGLANG_PATH=${SGLANG_PATH:-$SGLANG_ROOT/python}
PYTHON_BIN=${PYTHON_BIN:-python3}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$SCRIPT_DIR/a5_hicache_e2e_probe.py"

TP_SIZE=${TP_SIZE:-8}
# V4.1 A5 keeps all 64 query heads on every rank, which needs DP attention.
# Set ENABLE_DP=0 only for a model/backend that supports head-split TP.
ENABLE_DP=${ENABLE_DP:-1}
DP_SIZE=${DP_SIZE:-8}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.83}
HICACHE_RATIO=${HICACHE_RATIO:-1.2}
FILL_FACTOR=${FILL_FACTOR:-1.05}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-900}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/probe_out/e2e_target_$(date +%Y%m%d_%H%M%S)}

if [[ ! -f "$PROBE" ]]; then
  echo "ERROR: missing probe: $PROBE" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: missing model directory: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$SGLANG_PATH" ]]; then
  echo "ERROR: missing SGLANG_PATH: $SGLANG_PATH" >&2
  exit 1
fi
if ! awk -v r="$HICACHE_RATIO" -v f="$FILL_FACTOR" 'BEGIN {
  exit !(r > 1.0 && f > 1.0 && f < r)
}'; then
  echo "ERROR: need HICACHE_RATIO > 1.0 and 1.0 < FILL_FACTOR < HICACHE_RATIO" >&2
  echo "       got HICACHE_RATIO=$HICACHE_RATIO FILL_FACTOR=$FILL_FACTOR" >&2
  echo "       ratio 1.0 has no host headroom: filling device would evict A from host." >&2
  exit 1
fi

# ===== CPU / System =====
if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi
sysctl -w vm.swappiness=0 >/dev/null 2>&1 || true
sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING || true

# ===== Ascend Environment =====
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export TRANSFORMERS_VERBOSITY=error
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

export SGLANG_SET_CPU_AFFINITY=1
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

export HCCL_BUFFSIZE=400
export DEEPEP_HCCL_BUFFSIZE=1800

# ===== Ascend Toolkit =====
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then
  source /usr/local/memfabric_hybrid/set_env.sh
fi

# ===== Model / SGLang Environment =====
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# DSpark-specific switches are intentionally omitted.
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=150

export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

export SGLANG_DSV4_NATIVE_ATTN=0
export SGLANG_DSV4_DBG_ATTN=1
export SGLANG_SHALLOW_FUSED_ROPE_FQ4=0
export SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1
export SGLANG_COMPRESSOR_DECODE_USE_TRITON=1
export SGLANG_OPT_USE_HC_TRITON=0
export SGLANG_OPT_USE_HC_TILELANG=0

export SGLANG_DSV4_GRAPH_MAX_SEQ=131072

# ===== HiCache =====
export SGLANG_HICACHE_HOST_MEM_BACKEND=${SGLANG_HICACHE_HOST_MEM_BACKEND:-acl}
export SGLANG_ADMIN_API_KEY=${SGLANG_ADMIN_API_KEY:-sglang-admin}
export CANNBOTDSL_CACHE_DIR=${CANNBOTDSL_CACHE_DIR:-$HOME/.cache/cannbot_cache}

# The probe's server subprocess must import this checkout, not /sgl-workspace.
export PYTHONPATH="$SGLANG_PATH${PYTHONPATH:+:$PYTHONPATH}"

# ===== Preflight =====
mkdir -p "$OUT_DIR"
echo "=== A5 HiCache L2 target-only probe ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "SGLANG_PATH=$SGLANG_PATH"
echo "TP_SIZE=$TP_SIZE ENABLE_DP=$ENABLE_DP DP_SIZE=$DP_SIZE"
echo "MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC"
echo "HICACHE_RATIO=$HICACHE_RATIO FILL_FACTOR=$FILL_FACTOR"
echo "OUT_DIR=$OUT_DIR"
free -g || true

SGLANG_FILE=$("$PYTHON_BIN" -c 'import sglang; print(sglang.__file__)')
echo "sglang module: $SGLANG_FILE"
if [[ "$SGLANG_FILE" != "$SGLANG_PATH"/* ]]; then
  echo "ERROR: server would import $SGLANG_FILE, not $SGLANG_PATH" >&2
  echo "       fix PYTHONPATH or set SGLANG_ROOT to the checkout containing this branch" >&2
  exit 1
fi

# ===== Launch args =====
EXTRA_ARGS=(
  --attention-backend dsv4
  --reasoning-parser deepseek-v4
  --language-model-only
  --moe-a2a-backend deepep
  --deepep-mode auto
  --kv-cache-dtype auto
  --enable-metrics
  --watchdog-timeout 9000
  --max-running-requests 200
  --max-prefill-tokens 16384
  --chunked-prefill-size 4096
  --cuda-graph-bs-decode 1 2 4 8
)
if [[ "$ENABLE_DP" == "1" ]]; then
  EXTRA_ARGS+=(--enable-dp-attention --dp-size "$DP_SIZE" --enable-dp-lm-head)
fi

echo "=== running probe ==="
"$PYTHON_BIN" "$PROBE" \
  --model-path "$MODEL_PATH" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --hicache-ratio "$HICACHE_RATIO" \
  --fill-factor "$FILL_FACTOR" \
  --startup-timeout "$STARTUP_TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --sglang-path "$SGLANG_PATH" \
  --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}' \
  --extra-args "${EXTRA_ARGS[*]}" \
  --env SGLANG_DSV4_FP4_EXPERTS=True \
  --out-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/probe.log"

echo
echo "=== done ==="
echo "report: $OUT_DIR/report.json"
echo "logs:   $OUT_DIR"
