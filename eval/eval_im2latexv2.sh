#!/usr/bin/env bash
# --------------------------------------------------------------
# run_infer.sh – minimal wrapper for scripts/vllm_infer.py
# --------------------------------------------------------------
# Required:
#   -i / --input-dir    Path to input data
#   -o / --output-dir   Path to write results
#   -m / --model        Model name or checkpoint
# Optional:
#   -b / --batch_size   Samples per vLLM.generate() call (int, >0)
#
# Sets CUDA_VISIBLE_DEVICES to 0 by default (override by exporting
# CUDA_VISIBLE_DEVICES before calling the script).
# --------------------------------------------------------------
set -euo pipefail

INPUT_DIR="data/Im2LaTeXv2/prompts"
OUTPUT_DIR="data/Im2LaTeXv2/results"
MODEL=""
BATCH_SIZE=""

usage() {
  echo "Usage: $(basename "$0") -i <input-dir> -o <output-dir> -m <model> [-b <batch_size>]" >&2
  exit 1
}

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir)    INPUT_DIR="$2"; shift 2;;
    -o|--output-dir)   OUTPUT_DIR="$2"; shift 2;;
    -m|--model)        MODEL="$2"; shift 2;;
    -b|--batch_size)   BATCH_SIZE="$2"; shift 2;;
    -h|--help)         usage;;
    *)                 echo "[ERROR] Unknown option: $1" >&2; usage;;
  esac
done

[[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$MODEL" ]] && usage

# ---------- Validate batch size if provided ----------
if [[ -n "${BATCH_SIZE}" ]]; then
  if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -le 0 ]]; then
    echo "[ERROR] --batch_size must be a positive integer, got: '${BATCH_SIZE}'" >&2
    exit 1
  fi
fi

# ---------- Environment ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Using GPU(s): $CUDA_VISIBLE_DEVICES" >&2

# ---------- Build command ----------
cmd=(python scripts/vllm_infer.py
  --input-dir  "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --model      "$MODEL"
)

if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=(--batch-size "${BATCH_SIZE}")
fi

echo "[INFO] Command: ${cmd[*]}" >&2

# ---------- Run inference ----------
"${cmd[@]}"