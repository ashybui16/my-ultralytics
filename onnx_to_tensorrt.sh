#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <model.onnx> [model.engine]"
    exit 1
fi

ONNX="$1"
ENGINE="${2:-${ONNX%.*}.engine}"
ENGINE_DIR="$(dirname "$ENGINE")"
ENGINE_STEM="$(basename "${ENGINE%.*}")"
LOG_FILE="$ENGINE_DIR/${ENGINE_STEM}_log.txt"
BENCHMARK_OUT="$ENGINE_DIR/${ENGINE_STEM}_benchmark.json"
PROFILE_OUT="$ENGINE_DIR/${ENGINE_STEM}_profile.json"
LAYER_OUT="$ENGINE_DIR/${ENGINE_STEM}_layers.json"

[[ -f "$ONNX" ]] || {
    echo "ONNX not found: $ONNX" >&2
    exit 1
}

command -v trtexec >/dev/null || {
    echo "trtexec not found" >&2
    exit 1
}

mkdir -p "$ENGINE_DIR"

# Save all console output in the same directory as the engine file
exec > >(tee "$LOG_FILE") 2>&1

printf 'ONNX: %s\n' "$ONNX"
printf 'Engine: %s\n' "$ENGINE"
printf 'Log file: %s\n' "$LOG_FILE"
printf 'Benchmark output: %s\n' "$BENCHMARK_OUT"
printf 'Profile output: %s\n' "$PROFILE_OUT"
printf 'Layer info: %s\n' "$LAYER_OUT"

# Build TensorRT engine
trtexec \
    "--onnx=$ONNX" \
    "--saveEngine=$ENGINE" \
    --fp16 \
    --builderOptimizationLevel=5 \
    --memPoolSize=workspace:4096 \
    --skipInference

# Benchmark
trtexec \
    "--loadEngine=$ENGINE" \
    --shapes=images:1x3x640x640 \
    --warmUp=2000 \
    --duration=30 \
    --useCudaGraph \
    --useSpinWait \
    --noDataTransfers \
    --exportTimes="$BENCHMARK_OUT"

# Per-layer profiling
trtexec \
    "--loadEngine=$ENGINE" \
    --shapes=images:1x3x640x640 \
    --warmUp=1000 \
    --duration=10 \
    --useCudaGraph \
    --separateProfileRun \
    --dumpProfile \
    --dumpLayerInfo \
    --exportProfile="$PROFILE_OUT" \
    --exportLayerInfo="$LAYER_OUT"

printf '\nAll printed output has been saved to %s\n' "$LOG_FILE"
    