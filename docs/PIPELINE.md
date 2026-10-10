# Qwen detection, Python geometry, Apertus planning

The pipeline is:

```text
RGB image → Qwen labels, boxes, target pixels
          → Python segmentation, physical width, graspability, relations
          → compact scene graph → Apertus plan → submission
```

Qwen makes one detection request per image, with bounded retries for malformed
JSON. It never predicts relations, sizes, graspability or confidence. Python
assigns numbered IDs from labels. The Qwen-selected target pixel is preserved.

## Project layout

```text
run_pipeline.py               command-line entry point
prompts/
  qwen_system.txt             detection instructions
  apertus_system.txt          planning instructions
src/
  perception.py               Qwen detection contract, prompt and coordinate conversion
  compute_geometry.py         segmentation, minimum-area rectangle, width and relations
  vlm.py                      model loading and inference
  create_scene_graph.py       compact graph serialization
  apertus.py                  planning requests
  validation.py               JSON/schema structure only
  pipeline.py                 stage orchestration and scene-graph caching
apertus/
  start.sh                    vLLM server launcher
containers/
  vision.edf.toml             Qwen Clariden container configuration
  apertus.edf.toml            Apertus Clariden container configuration
```

Small helpers in `src/` handle schema loading, JSON parsing and submission formatting. `run.py` forwards to `run_pipeline.py` for
existing commands. `starter/` remains the supplied data loader and evaluator.

`RoboticsPipeline.run(..., perception_only=True)` creates or reuses scene graphs.
Graph loading/creation is shared and happens before the mode branch. With
`perception_only=False`, the `else` branch plans instructions using that graph.
The method returns the final results list, which is empty in perception-only mode.
Errors propagate to the caller; there are no diagnostic logs or failure reports.

## Detection and graph contracts

Qwen returns `objects`, each with:

```json
{
  "label": "yellow_gear",
  "bbox_2d": [1700, 660, 1800, 760],
  "target_pixel": [1747, 712]
}
```

Labels use color and category, such as `red_cube` and `white_plate`, without
patterns or decorations. Fruits use only their names, such as `orange` and
`banana`. Boxes use
half-open `[x1, y1, x2, y2]` coordinates. Pixel `[u, v]` means column then row,
with the origin at the top left. Qwen3-family normalized coordinates are converted
to original image pixels once: box minima round down, maxima up, target pixels
down. Qwen2.5 uses original pixels. There are no follow-up relation/point queries.

The final graph has exactly `objects` and `relations`:

```json
{
  "objects": [
    {"id": "yellow_gear_1", "label": "yellow_gear", "target_pixel": [1747, 712], "width_mm": 43.0, "graspable": true},
    {"id": "red_gear_1", "label": "red_gear", "target_pixel": [1775, 584]}
  ],
  "relations": [
    {"subject": "yellow_gear_1", "relation": "left", "object": "red_gear_1", "confidence": 0.2},
    {"subject": "yellow_gear_1", "relation": "below", "object": "red_gear_1", "confidence": 0.6}
  ]
}
```

This example shows two directional edges; their inverses are implicit and are
not stored. The numbers above illustrate the format, not a measured scene.
`width_mm` and `graspable` are omitted together if measurement fails. There is no
object confidence, image metadata, bounding box, support or occlusion in the
planner's graph. Full-precision width determines graspability.

## Geometry

1. Crop around the detected box, retaining a background margin. Resize large
   crops to at most 640 pixels on their longer axis to bound CPU work.
2. Run OpenCV GrabCut with the box interior as probable foreground, surrounding
   pixels as background, and the Qwen-chosen material pixel as definite foreground.
3. Select the connected foreground component containing that pixel and extract
   its external contour. No ground-truth annotations enter this computation.
4. Project the contour onto the horizontal plane through the selected pixel's
   measured 3D height when depth is available. For this dataset's 2D images, use
   the calibrated table plane at robot-base z = 4 mm.
5. Fit `cv2.minAreaRect` in physical coordinates. `width_mm` is its shorter side;
   `graspable` is exactly `width_mm <= 85`, independent of label.

An unmeasurable region, invalid sampling coordinates or missing calibration
leaves width and graspability unknown.
There is no guessed bounding-box width fallback. Geometry prerequisites protect
array access and projection; the scene-graph validator does not enforce them.

Segmentation can include neighboring material or miss part of a multicolored,
occluded or low-contrast object. The target selects a connected component, not
an independently verified object mask. In 2D, raised-object sizes are approximate
because projection uses table height. A width-based `graspable` flag describes
only gripper opening, not contact, accessibility or safety.

## Computed relations and confidence

An object's `inside` edge points to every detected tray whose half-open box
contains its target pixel. Its whole box need not fit. A tray is identified by
its label ending in `tray`; mats are not trays. Self-edges are excluded. There is
no redundant `inside_tray` boolean and no `on` relation.

Containment takes precedence: an object inside a tray gets no directional
relations to that tray. Relations between objects sharing a tray are still
computed normally.

For every other unordered pair, compare bounding-box edges along each axis.
Emit `left` or `right` only when the horizontal intervals do not overlap; emit
`above` or `below` only when the vertical intervals do not overlap. For example,
an object beside a tall tray can be right of it without also being above it.
Touching edges count as separated with zero confidence. These are image
directions, never physical stacking.

Each directional fact is stored once, relative to the pair's input order.
`A left B` also means `B right A`; the mirrored edge is omitted. Diagonally
separated objects can have one horizontal and one vertical relation.

Confidence is computed only for relations:

- Direction: `box_gap / (box_gap + mean_box_extent)` on the relevant axis.
- Inside: distance to the nearest tray-box edge divided by half the shorter tray
  extent, capped at 1. A target exactly on an included edge has confidence 0.

Scores are rounded to four decimals. They express geometric separation or margin,
not calibrated probabilities or Qwen certainty. A low score does not remove an
otherwise true box separation.

## Validation and planning

`schema/perception.schema.json` and `schema/scene_graph.schema.json` define the
two separate contracts. `validation.py` checks JSON representability, fields,
types, array lengths, enums and scalar format limits, with comments for each
check. It does not check unique object IDs, reference existence, coordinate
bounds, cycles, physical plausibility or consistency of graspability with width.
JSON parsing also rejects duplicate keys, incomplete JSON and non-finite numbers.

Qwen receives the detection schema in its prompt and generates text without a
custom token filter. Python parses the response and checks its structure afterward.
Malformed JSON or schema errors trigger a fresh request with error feedback, up to
`--qwen-validation-retries`. Thinking is disabled for detection. Geometry is
computed in Python after a valid detection response.

Apertus receives only the compact graph and instruction. The gripper opening is
fixed at 85 mm in the geometry calculation and Apertus system prompt. Its system
prompt uses measured graspability and computed relations, and treats absent
measurements as unknown. The graph cannot answer physical-support or
human-holding questions; the prompt asks for clarification when an instruction
depends on such facts. The plan schema checks output structure only. This code
writes challenge submissions and does not execute robot motion.

## Running

See [instructions.md](../instructions.md) for the full Clariden allocation,
environment, model startup and evaluation sequence. Keep the Apertus/vLLM
container environment separate from Qwen's environment and retain its CUDA
PyTorch installation. The optional `vision` dependencies in `pyproject.toml`
are needed for the default Qwen3 model; core geometry tests need only the base
dependencies.

Start Apertus, then run the complete pipeline. The launcher reserves a 32,768-token
context because explicit pairwise relations grow quadratically with object count;
set `APERTUS_MAX_MODEL_LEN` to adjust this. The official [Apertus model card](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)
documents support for longer contexts. Existing servers launched with an 8,192-token
limit may need restarting for crowded scenes:

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_BIN=/opt/venv/bin/vllm bash apertus/start.sh
# In another terminal on the same compute node:
CUDA_VISIBLE_DEVICES=1 .venv-clariden/bin/python -u run_pipeline.py \
  --data dataset --out outputs/results.json
```

Run detection and geometry without an Apertus server:

```sh
CUDA_VISIBLE_DEVICES=1 .venv-clariden/bin/python -u run_pipeline.py \
  --perception-only --scene scene_04
```

The default Qwen model is `Qwen/Qwen3-VL-32B-Instruct`; select another supported
checkpoint with `--qwen-model`. The default generation budget is 4096 tokens with
two format retries. `--limit` counts instructions and only applies to full runs.
`--attention`, `--timeout` and `--no-constrained-decoding` retain their existing
roles; the last option affects Apertus only.

The pipeline loads or creates one graph for each scene and passes it to every
instruction for that scene. Accepted graphs are saved in `scene_graphs/` beside
the submission path and reused across runs by scene name.

## Artifacts and tests

Both modes save accepted scene graphs and a matching `<scene>.detections.png`
beside each newly generated graph. The original-resolution overlay shows Qwen's
bounding boxes, object IDs and target-point crosses for visual inspection.
It does not calculate an accuracy score or draw the fitted measurement rectangle.
Full runs also write the final submission at the requested output path.

Cached graphs and their existing overlays are reused. Older cached graphs do not
contain bounding boxes, so their overlays cannot be reconstructed. Use a fresh
`--out` parent directory to generate detections and overlays for those scenes.

After format retries are exhausted, an error stops the run and is raised directly.
Existing diagnostic files from earlier runs are not deleted.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests use synthetic geometry and stub model responses. They cover rotated
segmentation, target-depth projection, the 85 mm boundary, unknown measurements,
relation directions, tray boundaries, schema-only validation, coordinate
conversion, retries, compact planner input and graph reuse across instructions. Real model accuracy
requires running the models and comparing detections with the source images.
