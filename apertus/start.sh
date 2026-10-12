#!/usr/bin/env bash
set -euo pipefail
exec "${VLLM_BIN:-vllm}" serve swiss-ai/Apertus-v1.5-70B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size "${APERTUS_TENSOR_PARALLEL_SIZE:-2}" \
  --max-model-len "${APERTUS_MAX_MODEL_LEN:-32768}" --gpu-memory-utilization 0.9 \
  --chat-template-content-format string "$@"
