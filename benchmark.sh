#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <model.engine> [shape]"
    echo "Example: $0 /path/to/model.engine"
    echo "Example: $0 /path/to/model.engine images:1x3x640x640"
    exit 1
fi

ENGINE="$1"
SHAPE="${2:-images:1x3x640x640}"

[[ -f "$ENGINE" ]] || {
    echo "Engine not found: $ENGINE" >&2
    exit 1
}

command -v trtexec >/dev/null || {
    echo "trtexec not found in PATH" >&2
    exit 1
}

DIR="$(dirname "$ENGINE")"
BASE="$(basename "$ENGINE")"
STEM="${BASE%.*}"

BENCHMARK_OUT="$DIR/${STEM}_benchmark.json"
PROFILE_OUT="$DIR/${STEM}_profile.json"
LAYER_OUT="$DIR/${STEM}_layers.json"
LOG_FILE="$DIR/${STEM}_log.txt"

# Save all console output to a text log in the same directory as the engine
exec > >(tee "$LOG_FILE") 2>&1

printf 'Engine: %s\n' "$ENGINE"
printf 'Shape: %s\n' "$SHAPE"
printf 'Log file: %s\n' "$LOG_FILE"
printf '\nBenchmark output: %s\n' "$BENCHMARK_OUT"
printf 'Profile output: %s\n' "$PROFILE_OUT"
printf 'Layer info: %s\n' "$LAYER_OUT"

# Benchmark: timing stats
trtexec \
    --loadEngine="$ENGINE" \
    --shapes="$SHAPE" \
    --warmUp=2000 \
    --duration=30 \
    --useCudaGraph \
    --useSpinWait \
    --noDataTransfers \
    --exportTimes="$BENCHMARK_OUT"

# Per-layer profiling
trtexec \
    --loadEngine="$ENGINE" \
    --shapes="$SHAPE" \
    --warmUp=1000 \
    --duration=10 \
    --useCudaGraph \
    --separateProfileRun \
    --dumpProfile \
    --dumpLayerInfo \
    --exportProfile="$PROFILE_OUT" \
    --exportLayerInfo="$LAYER_OUT"

printf '\nAll printed output has been saved to %s\n' "$LOG_FILE"
