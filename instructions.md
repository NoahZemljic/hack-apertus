# Clariden: full run, planning replay and benchmark comparison

The supported pipeline is Apertus extraction → Qwen category discovery and
verification → Grounding DINO → SAM 2 → geometry → Apertus planning.
`run_grounded.py` is its only entry point.
The full build, cache and deployment instructions live in
[docs/CLARIDEN.md](docs/CLARIDEN.md); prepare that image and its model caches first.

## 1. Open the four-GPU container

From a Clariden login node, using your mounted checkout:

```bash
cd "$HOME/hack-apertus" || exit 1
srun -A g239 --partition=debug --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --cpus-per-task=64 --time=01:30:00 \
  --environment=./containers/clariden.edf.toml --pty bash
```

All remaining commands run inside this container. It already provides the
matching engine and dependencies; do not create a second Qwen environment or
replace the image's Torch, Transformers or vLLM packages.

## 2. Run every stage and evaluate

```bash
cd "$HOME/hack-apertus" || exit 1
mkdir -p "$HACK_APERTUS_OUTPUT"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/full-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --debug-visualizations \
  --out "$run_dir/results.json"

python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

No `--limit` means all scenes and instructions. The launcher starts the local
Apertus 70B and Qwen3-VL-32B servers, runs DINO/SAM on the fourth GPU, then stops
the servers. Each run gets a separate output directory. The mounted checkout's
code, schemas and prompts are used immediately; rebuild the image when changing
dependencies.

The output folder contains the intermediate detections, masks and geometry,
one planner graph per scene, the submission and the exact planning requests.
Review `run_manifest.json`, `planning_policy.txt`, `results.raw.jsonl` and
`report.json` to connect a score to the code and prompt that produced it.
Only the evaluator reads challenge ground truth.

## 3. Replan from a saved run

To test planning changes while retaining the previous detections and masks:

```bash
cd "$HOME/hack-apertus" || exit 1
previous_run="$HACK_APERTUS_OUTPUT/full-<previous-run-name>"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/replan-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --geometry-input "$previous_run/geometry_results.json" \
  --out "$run_dir/results.json"

python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

Keep the previous mask files in place. This rebuilds the planner graphs and
spatial context from saved geometry and runs Apertus only. Repeat the previous
run's graph options, such as `--no-support-candidates`, when holding those inputs
constant. Planning makes one decision per instruction, with one retry only for
malformed JSON or schema structure.

## 4. Copy the artifacts for inspection

```bash
mkdir -p "$HOME/hack-apertus/outputs"
cp -r --no-preserve=mode,ownership "$run_dir" "$HOME/hack-apertus/outputs/"
```

This avoids preserving permissions and ACLs that some container home mounts
reject. The original run remains in scratch.

## Optional: compare with the supplied benchmark

`baseline/baseline.py` is the challenge's original Apertus-only benchmark. It
is retained for score comparison and is separate from the supported pipeline.
For its model setup and usage, see [JURY_README.md](JURY_README.md#quickstart-15-minutes).
The benchmark's `--mock` option checks file loading and scoring without models:

```bash
benchmark_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/benchmark-XXXXXX")" || exit 1
python baseline/baseline.py --mock --data "$HACK_APERTUS_DATA" \
  --out "$benchmark_dir/results.json"
python starter/evaluate.py "$benchmark_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$benchmark_dir/report.json"
```

A mock run is a setup check, not a model score. To compare two actual reports:

```bash
python - "$run_dir/report.json" "$benchmark_dir/report.json" <<'PY'
import json
import sys
from pathlib import Path

print(f'{"Run":<12} {"Grounding /35":>14} {"Safety /30":>12} {"Total /65":>12}')
for name, filename in zip(("pipeline", "benchmark"), sys.argv[1:]):
    report = json.loads(Path(filename).read_text())
    print(f'{name:<12} {35 * report["grounding"]:>14.1f} '
          f'{30 * report["safety"]:>12.1f} {report["automatic_score"]:>12.1f}')
PY
```

Replace `benchmark_dir` with the directory of the actual benchmark run before
making that comparison. These are end-to-end challenge scores; inspect the
saved detection and graph overlays separately for perception quality.
