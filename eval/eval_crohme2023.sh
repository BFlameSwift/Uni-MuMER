#!/usr/bin/env bash
# --------------------------------------------------------------
# run_infer.sh – minimal wrapper for scripts/infer2.py
# --------------------------------------------------------------
# Accepts exactly three required arguments:
#   -i / --input-dir   Path to input data
#   -o / --output-dir  Path to write results
#   -m / --model       Model name or checkpoint
#
# Sets CUDA_VISIBLE_DEVICES to 0 by default (override by exporting
# CUDA_VISIBLE_DEVICES before calling the script).
# --------------------------------------------------------------
set -euo pipefail

INPUT_DIR="data/CROHME2023/prompts"
OUTPUT_DIR="data/CROHME2023/results"
MODEL=""

usage() {
  echo "Usage: $(basename \"$0\") -i <input-dir> -o <output-dir> -m <model>" >&2
  exit 1
}

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir)    INPUT_DIR="$2"; shift 2;;
    -o|--output-dir)   OUTPUT_DIR="$2"; shift 2;;
    -m|--model)        MODEL="$2"; shift 2;;
    -h|--help)         usage;;
    *)                 echo "[ERROR] Unknown option: $1" >&2; usage;;
  esac
done

[[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$MODEL" ]] && usage

# ---------- Environment ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Using GPU(s): $CUDA_VISIBLE_DEVICES" >&2

# ---------- Run inference ----------
python scripts/vllm_infer.py \
  --input-dir  "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model      "$MODEL"