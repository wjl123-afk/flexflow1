#!/bin/bash
set -euo pipefail

# 可通过环境变量覆盖这些默认参数
BASE_URL="${BASE_URL:-http://127.0.0.1:8015}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/llama3.1-8B}"
RESULT_DIR="${RESULT_DIR:-/workspace/gpu_model_vllm_test/RTX_4090_24GB_PCIe/Llama-3_1-8B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
METRIC_PERCENTILE="${METRIC_PERCENTILE:-50,90,99}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
DATASET_NAME="${DATASET_NAME:-random}"

INPUT_LENGTHS=(128 512 1024 2048)
OUTPUT_LENGTHS=(128 512 1024 2048)

for input_len in "${INPUT_LENGTHS[@]}"; do
  for output_len in "${OUTPUT_LENGTHS[@]}"; do
    result_filename="Llama-3_1-8B-Instruct__RTX_4090_24GB_PCIe__TP1_prompt${input_len}_output${output_len}.json"
    echo ">>> 执行 input=${input_len}, output=${output_len}"
    vllm bench serve \
      --backend vllm \
      --base-url "${BASE_URL}" \
      --model "${MODEL_DIR}" \
      --dataset-name "${DATASET_NAME}" \
      --random-input-len "${input_len}" \
      --random-output-len "${output_len}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --metric-percentile "${METRIC_PERCENTILE}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --top-k "${TOP_K}" \
      --save-result \
      --result-dir "${RESULT_DIR}" \
      --result-filename "${result_filename}"
  done
done

