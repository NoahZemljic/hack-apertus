#!/usr/bin/env bash
set -euo pipefail
exec "${VLLM_BIN:-vllm}" serve "${APERTUS_MODEL:-swiss-ai/Apertus-v1.5-8B}" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len "${APERTUS_MAX_MODEL_LEN:-32768}" --gpu-memory-utilization 0.8 \
  --chat-template-content-format string "$@"
