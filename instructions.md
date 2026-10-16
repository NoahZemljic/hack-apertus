# Clariden: full run, planning replay and benchmark comparison

The supported pipeline is Apertus extraction → Qwen3-VL-32B boxes →
SAM 2 → geometry + separate Qwen stacking → Apertus planning.
`run_grounded.py` is its only entry point.
Qwen3-VL-32B finds all visible object instances and their boxes in one request
per scene, then separately identifies directed physical support relationships
between candidate objects to enrich the graph.
The full build, cache and deployment instructions live in
[README.md](README.md#getting-started-on-clariden); prepare that image and its model caches first.

## 1. Open the four-GPU container

From a Clariden login node, using your mounted checkout. Replace
`<project_account>` with your Slurm project account before running the command:

```bash
cd "$HOME/hack-apertus" || exit 1
srun -A <project_account> --partition=normal --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --cpus-per-task=64 --time=01:30:00 \
  --environment=./containers/clariden.edf.toml --pty bash
```

All remaining commands run inside this container. It already provides the
matching engine and dependencies; do not create a second Qwen environment or
replace the image's Torch, Transformers or vLLM packages.

## 2. Run the dataset or held-out examples

Choose one of the following complete run blocks inside the container. Both use
Qwen boxes and the separate Qwen stacking step, and save each run in a fresh
directory under `$HOME/hack-apertus/outputs`. Pass `--data` explicitly to select the examples.

### Dataset examples: run and evaluate

Process all 10 public scenes and 50 instructions, then score against the supplied
ground truth:

```bash
cd "$HOME/hack-apertus" || exit 1
mkdir -p "$HACK_APERTUS_OUTPUT"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/dataset-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 QWEN_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --data "$HOME/hack-apertus/dataset" \
  --stacking-source qwen \
  --debug-visualizations \
  --out "$run_dir/results.json"
```

### Held-out examples: run and save the submission

Process all 5 held-out scenes and 40 instructions:

```bash
cd "$HOME/hack-apertus" || exit 1
mkdir -p "$HACK_APERTUS_OUTPUT"
run_dir="$(mktemp -d "$HACK_APERTUS_OUTPUT/heldout-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")" || exit 1

time APERTUS_MAX_MODEL_LEN=16384 QWEN_MAX_MODEL_LEN=16384 bash containers/start_pipeline.sh -- \
  --data "$HOME/hack-apertus/heldout" \
  --stacking-source qwen \
  --debug-visualizations \
  --out "$run_dir/results_heldout.json"
```

Submit `$run_dir/results_heldout.json`. The released held-out folder has no
ground-truth answers or labeled masks, so local scoring is unavailable. The
organizers evaluate it with the same script using their private annotations.
The artifacts described in section 4 are available after either run block.

No `--limit` means all scenes and instructions. The launcher starts the local
Apertus 70B and Qwen3-VL-32B servers, runs SAM on the fourth GPU, then stops the
servers before evaluating the final submission. The mounted checkout's code,
schemas and prompts are used immediately; rebuild the image when changing dependencies.

The output folder contains intermediate detections, masks and geometry, one
planner graph per scene, the submission and exact planning requests. Review
`run_manifest.json`, `planning_policy.txt` and `results*.raw.jsonl`; dataset runs
also produce `report.json` in the launcher's final evaluation step, after model
shutdown. `evaluation.json` records whether scoring completed, failed, or was
skipped because annotations are unavailable.
Only the evaluator reads challenge ground truth.

### Qwen perception

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
Qwen boxes pass through cleanup, SAM masks and picking points, geometry, scene
graphs and the Apertus planning policy. Detection uses the existing Qwen server.
The command gives Qwen a 16384-token context to leave room for the full image,
detection prompt and up to 4096 output tokens for the instance boxes.
The detector prompt now follows `feature/qwen-only`'s object and box guidance:
region-by-region scanning, separate repeated instances, low-contrast tray contents,
snake_case labels and fruit names without color prefixes. Labels are normalized
before SAM and graph construction; the flat black placement surface retains
the instruction name `black tray`. Picking points still come from SAM.

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
  --out "$run_dir/results.json"
```

Keep the previous mask PNGs and original dataset images accessible. If the run is
still in scratch, set `previous_run` to that scratch folder instead. For copied
checkpoints whose original scratch paths are unavailable, refresh reuses the
accompanying `segmentation/<scene>/<object-id>.png` masks. It starts the existing
Qwen server and Apertus; it needs no SAM model weights.
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
  --out "$run_dir/results.json"
```

Keep the previous mask files in place. This rebuilds the planner graphs and
spatial context from saved geometry and runs Apertus only. Saved Qwen support
relations are retained without a new visual call. Repeat the previous
run's graph options, such as `--no-support-candidates`, when holding those inputs
constant; that option disables legacy geometry guesses, not saved Qwen support.
Planning makes one decision per instruction, with one retry only for
malformed JSON or schema structure.

## 4. Inspect the saved artifacts

Run artifacts are already saved under `$HOME/hack-apertus/outputs`.
Inspect the submission, reports and intermediate files in `$run_dir`.

## Optional: compare with the supplied benchmark

`baseline/baseline.py` is the challenge's original Apertus-only benchmark. It
is retained for score comparison and is separate from the supported pipeline.
For its usage, run `python baseline/baseline.py --help`.
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
