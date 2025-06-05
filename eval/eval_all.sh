#!/usr/bin/env bash
# eval_all.sh – unified evaluation entry point

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1      # Change GPU ID here if needed

usage() {
  echo "Usage: $0 -m MODEL" >&2
  exit 1
}

MODEL=""

# ---------- Parse command-line options ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    -h|--help)  usage ;;
    *)          echo "[ERROR] Unknown option: $1" >&2; usage ;;
  esac
done

[[ -z "$MODEL" ]] && { echo "[ERROR] -m/--model is required." >&2; usage; }

# ---------- Run all evaluations ----------
scripts=(
  eval/eval_crohme.sh
  eval/eval_crohme2023.sh
  eval/eval_hme100k.sh
  eval/eval_mathwriting.sh
  eval/eval_im2latexv2.sh
  eval/eval_MNE.sh
)

dirs=(
  data/CROHME
  data/CROHME2023
  data/HME100K
  data/MathWriting
  data/Im2LaTeXv2
  data/MNE
)

# ---------- verify that every dataset folder exists ----------
for d in "${dirs[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "[ERROR] Directory '$d' not found - aborting." >&2
    exit 1
  fi
done

# ---------- Run each evaluation script ----------
echo "Starting evaluations with model: $MODEL"
echo "----------------------------------------"
for i in "${!scripts[@]}"; do
  echo "command: bash ${scripts[i]} -m $MODEL -i ${dirs[i]}/prompts -o ${dirs[i]}/results"
  bash "${scripts[i]}" -m "$MODEL" -i "${dirs[i]}/prompts" -o "${dirs[i]}/results"

  echo "[INFO] Finished evaluation for ${dirs[i]}"
  echo "----------------------------------------"
  echo
done