#!/usr/bin/env bash
# eval_all.sh – unified evaluation entry point

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1      # Change GPU ID here if needed

usage() {
  echo "Usage: $0 -m MODEL" >&2
  echo "  -m, --model    Path to the model to evaluate (required)" >&2
  echo "  -h, --help     Show this help message" >&2
  echo " -s --suffix    Suffix for the output directory (optional, default: 'results')" >&2
  echo "  -b, --batch_size    Batch size to forward to eval scripts (optional)" >&2
  exit 1
}

MODEL=""
BATCH_SIZE=""

# ---------- Parse command-line options ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    -h|--help)  usage ;;
    -s|--suffix) SUFFIX="$2"; shift 2 ;;
    -b|--batch_size)  BATCH_SIZE="$2"; shift 2 ;;
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
  out_dir="${dirs[i]}/results${SUFFIX:+_"$SUFFIX"}"   # output directory with optional suffix
  mkdir -p "$out_dir"       # ensure output directory exists

  cmd=(python scripts/vllm_infer.py
    --input-dir  "${dirs[i]}/prompts"
    --output-dir "${out_dir}"
    --model      "${MODEL}"
  )

  if [[ -n "${BATCH_SIZE}" ]]; then
    cmd+=(--batch-size "${BATCH_SIZE}")
  fi
  echo "command: ${cmd[*]}"
  "${cmd[@]}"
  # bash "${scripts[i]}" -m "$MODEL" -i "${dirs[i]}/prompts" -o "${out_dir}" 

  echo "[INFO] Finished evaluation for ${dirs[i]}"
  echo "----------------------------------------"
  echo
done