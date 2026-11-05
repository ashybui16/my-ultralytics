#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <model.pt> [imgsz] [batch]"
    exit 1
fi

MODEL="$1"
IMGSZ="${2:-640}"
BATCH="${3:-1}"

[[ -f "$MODEL" ]] || { echo "Không tìm thấy model: $MODEL" >&2; exit 1; }

command -v yolo >/dev/null || { echo "Không tìm thấy lệnh yolo" >&2; exit 1; }

yolo export \
    "model=$MODEL" \
    format=onnx \
    opset=17 \
    "imgsz=$IMGSZ" \
    "batch=$BATCH"