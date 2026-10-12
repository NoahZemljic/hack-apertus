# Clariden: allocate, run all scenes, compare with baseline

Run each block in order and stop if a command fails. Assumes the repo is at
`/users/noahzemljic/hack-apertus` and you are already on a Clariden login node.
This comparison runs the complete Qwen + Apertus pipeline, not perception alone.

## 1. Login node: update the repo and enter a compute-node container

```bash
cd /users/noahzemljic/hack-apertus
git pull --ff-only

srun --account=g239 --partition=debug --nodes=1 --ntasks=1 \
  --time=01:30:00 --job-name=hack-apertus \
  --environment=./containers/apertus.edf.toml --pty bash
```

Debug allows 90 minutes on one node, including startup/downloads. Keep this
terminal open. All remaining commands run inside this same compute-node shell.
For a longer comparison, request `--partition=normal --time=04:00:00` instead.
[CSCS partition limits](https://docs.cscs.ch/clusters/clariden/#partitions)

## 2. Compute node: prepare the environment

Keep Apertus/vLLM in `/opt/venv`; use `.venv-clariden` for Qwen and evaluation.
Do not install Transformers 4 into `/opt/venv` or reinstall PyTorch.

```bash
cd /users/noahzemljic/hack-apertus
export HF_HOME="${SCRATCH:?SCRATCH is not set}/hack-apertus/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
set -o pipefail

if [ ! -x .venv-clariden/bin/python ]; then
  /opt/venv/bin/python -m venv --system-site-packages .venv-clariden
fi

# Expose the container's CUDA PyTorch packages to the Qwen environment
CONTAINER_SITE=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
VISION_SITE=$(.venv-clariden/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$CONTAINER_SITE" > "$VISION_SITE/container-packages.pth"

.venv-clariden/bin/python -m pip install \
  "transformers>=5.18,<6" "qwen-vl-utils>=0.0.8,<1" \
  "jsonschema>=4.23,<5" \
  "numpy>=2,<3" "opencv-python-headless>=4.10,<5"

CUDA_VISIBLE_DEVICES=2 .venv-clariden/bin/python -c \
'import torch, torchvision, cv2; from transformers import AutoModelForImageTextToText; from qwen_vl_utils import process_vision_info; print("GPU:", torch.cuda.get_device_name(0)); print("CUDA:", torch.ones(1, device="cuda").item())'
```

If model downloads require authentication, run `/opt/venv/bin/hf auth login`
interactively once. The 70B model has a separate access agreement on its Hugging Face page.

## 3. Start Apertus on GPUs 0 and 1 and wait for readiness

Run this once per allocation. The timestamp keeps previous outputs intact.

```bash
RUN_DIRECTORY="$HOME/hack-apertus/outputs/comparison-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIRECTORY/pipeline" "$RUN_DIRECTORY/baseline"
git rev-parse HEAD > "$RUN_DIRECTORY/commit.txt"
printf 'Outputs: %s\n' "$RUN_DIRECTORY"

CUDA_VISIBLE_DEVICES=0,1 VLLM_BIN=/opt/venv/bin/vllm bash apertus/start.sh \
  > "$RUN_DIRECTORY/apertus.log" 2>&1 &
APERTUS_PID=$!

curl --fail --silent --show-error --retry 120 --retry-connrefused \
  --retry-delay 5 --max-time 5 --output /dev/null \
  --write-out 'HTTP %{http_code}\n' http://127.0.0.1:8000/health
```

Proceed only after `HTTP 200`. Startup can take several minutes. If it fails,
inspect `tail -n 80 "$RUN_DIRECTORY/apertus.log"`. `localhost` must refer to this
compute node, not a login node or a different allocation.

## 4. Run Qwen + Apertus across all 10 scenes / 50 instructions

Use no `--limit` to process all instructions. Each scene graph is loaded or created
once and shared by that scene's instructions. Graphs are saved under
`$RUN_DIRECTORY/pipeline/scene_graphs` and reused if that output directory is used again.

```bash
CUDA_VISIBLE_DEVICES=2 .venv-clariden/bin/python -u run_pipeline.py \
  --data dataset \
  --out "$RUN_DIRECTORY/pipeline/results.json" \
  --qwen-model Qwen/Qwen3-VL-32B-Instruct \
  --apertus-model swiss-ai/Apertus-v1.5-70B \
  --apertus-base-url http://127.0.0.1:8000/v1 \
  --attention sdpa --timeout 300 \
  2>&1 | tee "$RUN_DIRECTORY/pipeline/run.log"
```

If perception fails, the command stops with the error. Final results are written
only after the whole run completes.

## 5. Run the provided baseline against the same Apertus server

```bash
.venv-clariden/bin/python -u baseline/baseline.py \
  --data dataset --out "$RUN_DIRECTORY/baseline/results.json" \
  --model swiss-ai/Apertus-v1.5-70B \
  --base-url http://127.0.0.1:8000/v1 --timeout 300 \
  2>&1 | tee "$RUN_DIRECTORY/baseline/run.log"
```

This is the supplied direct-image baseline, without the Qwen scene graph.

## 6. Score both runs and print the comparison

```bash
.venv-clariden/bin/python starter/evaluate.py \
  "$RUN_DIRECTORY/pipeline/results.json" --data dataset \
  --report "$RUN_DIRECTORY/pipeline/report.json" \
  | tee "$RUN_DIRECTORY/pipeline/score.txt"

.venv-clariden/bin/python starter/evaluate.py \
  "$RUN_DIRECTORY/baseline/results.json" --data dataset \
  --report "$RUN_DIRECTORY/baseline/report.json" \
  | tee "$RUN_DIRECTORY/baseline/score.txt"

.venv-clariden/bin/python - "$RUN_DIRECTORY" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
scores = {}
print(f'{"Run":<12} {"Grounding /35":>14} {"Safety /30":>12} {"Total /65":>12}')
for name in ("pipeline", "baseline"):
    report = json.loads((directory / name / "report.json").read_text())
    scores[name] = report["automatic_score"]
    print(f'{name:<12} {35 * report["grounding"]:>14.1f} {30 * report["safety"]:>12.1f} {scores[name]:>12.1f}')
print(f'Pipeline minus baseline: {scores["pipeline"] - scores["baseline"]:+.1f} points')
PY
```

These are end-to-end challenge scores, not standalone scene-graph accuracy.
Compare saved scene graphs with the source images for perception quality.

## 7. Release the allocation when finished

```bash
kill "$APERTUS_PID"
exit
```

No inference, scoring, or baseline commands above execute robot motion.
