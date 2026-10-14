# Clariden: full run, planning replay and benchmark comparison

The supported pipeline is Apertus extraction → selectable object detector →
SAM 2 → geometry + separate Qwen stacking → Apertus planning.
`run_grounded.py` is its only entry point.
`--box-detector dino` (the default) uses Qwen category discovery and verification
followed by Grounding DINO boxes. `--box-detector qwen` asks the existing Qwen
server for all visible object instances and their boxes in one request per scene.
Both detector modes then ask Qwen separately for directed physical support
relationships between candidate objects; this enriches the existing graph.
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

time APERTUS_MAX_MODEL_LEN=16384 QWEN_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --box-detector dino \
  --debug-visualizations \
  --out "$run_dir/results.json" &&
python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

No `--limit` means all scenes and instructions. The launcher starts the local
Apertus 70B and Qwen3-VL-32B servers, runs SAM (and DINO when selected) on the
fourth GPU, then stops the servers. Each run gets a separate output directory. The mounted checkout's
code, schemas and prompts are used immediately; rebuild the image when changing
dependencies.

The output folder contains the intermediate detections, masks and geometry,
one planner graph per scene, the submission and the exact planning requests.
Review `run_manifest.json`, `planning_policy.txt`, `results.raw.jsonl` and
`report.json` to connect a score to the code and prompt that produced it.
Only the evaluator reads challenge ground truth.

### Compare the Qwen detector

Inside the same four-GPU container, run the complete pipeline with Qwen boxes:

```bash
cd "$HOME/hack-apertus" || exit 1
mkdir -p "$HACK_APERTUS_OUTPUT"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/qwen-boxes-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 QWEN_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --box-detector qwen --debug-visualizations \
  --out "$run_dir/results.json" &&
python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

This backend combines discovery and localization. Apertus's object terms guide
attention; they do not restrict Qwen's inventory. Each visible instance gets one
descriptive label and one normalized 0–1000 box, converted once to image pixels.
There is one detection pass, with one retry for malformed output by default
(`--qwen-box-retries 1`); no inventory review is added. A separate stacking request
uses the existing objects, IDs and image to identify physical supports; it does
not change boxes or rerun discovery. `--stacking-source qwen` is the default.
There is one batched request per scene with touching/overlapping candidate pairs,
and one format retry by default (`--qwen-stacking-retries 1`). An object can rest
on multiple supports. To compare with the previous geometry-only approach, run
a fresh run with `--stacking-source geometry`.
Qwen provides no calibrated detection score, so its confidence is saved as `null`.
Both detectors share box cleanup, SAM masks and picking points, geometry, scene
graphs and the Apertus planning policy. The Qwen option does not load or require
DINO weights. It uses the existing server and dependencies, so these source changes
do not require an image rebuild when running from the mounted checkout.
The command gives Qwen a 16384-token context to leave room for the full image,
detection prompt and up to 4096 output tokens for the instance boxes.
The detector prompt now follows `feature/qwen-only`'s object and box guidance:
region-by-region scanning, separate repeated instances, low-contrast tray contents,
snake_case labels and fruit names without color prefixes. Labels are normalized
before SAM and graph construction; the flat black placement surface retains
the instruction name `black tray`. Picking points still come from SAM.

Compare against the latest DINO discovery run (39.35/65), if copied locally:

```bash
previous_report="$HOME/hack-apertus/outputs/discovery-20261014T191752-JeM4C7/report.json"
python - "$previous_report" "$run_dir/report.json" <<'PY'
import json
import sys
from pathlib import Path

for name, filename in zip(("DINO", "Qwen"), sys.argv[1:]):
    report = json.loads(Path(filename).read_text())
    refusals = sum(item["expected"] == "plan" and item["answer"] == "refuse"
                   for item in report["instructions"])
    print(f'{name}: grounding {35 * report["grounding"]:.2f}/35, '
          f'safety {30 * report["safety"]:.2f}/30, '
          f'total {report["automatic_score"]:.2f}/65, false refusals {refusals}')
PY
```

For a comparison using the current checkout for both detectors, repeat the fresh
run block with `--box-detector dino` and a separate `dino-boxes-...` directory.
Changing the detector requires a full run; `--geometry-input` skips detection.

## 3. Add stacking to the latest saved run and replan

This keeps the latest Qwen boxes and SAM masks, calls Qwen only for support
relationships, then runs Apertus with the revised planning prompt:

```bash
cd "$HOME/hack-apertus" || exit 1
previous_run="$HOME/hack-apertus/outputs/qwen-boxes-20261014T203039-jbzKBv"
mkdir -p "$HACK_APERTUS_OUTPUT"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/qwen-stacking-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 QWEN_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --geometry-input "$previous_run/geometry_results.json" \
  --refresh-stacking --stacking-source qwen \
  --out "$run_dir/results.json" &&
python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

Keep the previous mask PNGs and original dataset images accessible. If the run is
still in scratch, set `previous_run` to that scratch folder instead. For copied
checkpoints whose original scratch paths are unavailable, refresh reuses the
accompanying `segmentation/<scene>/<object-id>.png` masks. It starts the existing
Qwen server and Apertus; it needs neither DINO nor SAM model weights.
Inspect `stacking_relations.json`, `qwen_stacking.raw.json` and the rebuilt graphs
to see the added directed links. The new geometry saves those observations for
future replay. These source and prompt changes use the mounted checkout and do
not require an image rebuild.

### Test only the planning prompt

To test planning changes while retaining the previous detections and masks:

```bash
cd "$HOME/hack-apertus" || exit 1
previous_run="$HACK_APERTUS_OUTPUT/full-<previous-run-name>"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/replan-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --geometry-input "$previous_run/geometry_results.json" \
  --out "$run_dir/results.json" &&
python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

Keep the previous mask files in place. This rebuilds the planner graphs and
spatial context from saved geometry and runs Apertus only. Saved Qwen support
relations are retained without a new visual call. Repeat the previous
run's graph options, such as `--no-support-candidates`, when holding those inputs
constant; that option disables legacy geometry guesses, not saved Qwen support.
Planning makes one decision per instruction, with one retry only for
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
