# Grounded perception and Apertus planning

`run_grounded.py` runs the full separate pipeline:

1. Apertus extracts object descriptions from each instruction.
2. Descriptions are combined per scene and passed to Qwen with the image.
3. Qwen discovers visible objects and verifies its list in a second pass.
4. Grounding DINO detects those verified categories; deterministic cleanup merges duplicates.
5. SAM segments cleaned boxes and selects foreground safe pixels.
6. Geometry calculates spatial relations and selected support hypotheses.
7. The deterministic builder constructs and strictly validates the planner graph.
8. Apertus receives that graph and each instruction to produce the submission.

`run_geometry.py` is a compatibility alias for this same full pipeline. The Qwen
graph pipeline in `run.py` keeps its defaults and output locations.

Run with PyTorch and the vision dependencies installed (the existing GPU setup
in [PIPELINE.md](PIPELINE.md) applies). Start Apertus on port 8000 and your Qwen
vision server on port 8001, with served model name `qwen-vl`, or override their
URLs and model names below. DINO and SAM load locally. This entrypoint uses the
standard library for server calls and does not require the OpenAI Python SDK.

Run and evaluate:

```sh
uv run --extra vision run_grounded.py --dino-device cuda --sam-device cuda
uv run starter/evaluate.py outputs/grounded_pipeline/results.json --data dataset --report outputs/grounded_pipeline/report.json
```

To inspect existing geometry without any model calls:

```sh
uv run run_grounded.py --geometry-input geometry_results.json --build-only
```

Useful options:

```sh
uv run --extra vision run_grounded.py --limit 5
uv run --extra vision run_grounded.py --scene scene_06
uv run --extra vision run_grounded.py --qwen-base-url http://localhost:8001/v1 --qwen-model qwen-vl
uv run --extra vision run_grounded.py --build-only
uv run run_grounded.py --geometry-input geometry_results.json
uv run run_grounded.py --geometry-input geometry_results.json --no-support-candidates --out outputs/grounded_no_support/results.json
```

The default starts from instructions and images each time. `--geometry-input`
(also `--input`) explicitly skips all upstream perception and reuses saved
geometry and masks. Relative paths in that file resolve against its directory,
or `--base-dir`. Fresh runs write absolute image/mask paths so their generated
geometry can be replayed from another working directory.

Only Qwen's verified categories reach DINO by default. `--query-source
grounding_queries` combines them with Apertus-extracted descriptions as an
explicit experiment. An instruction mentioning an absent object therefore
does not automatically add it to the default detector vocabulary.

`--limit` caps final planning and skips perception for later scenes. Every
instruction in each included scene is still extracted to build a shared scene
inventory. `--build-only` skips final planning; fresh perception still calls
Apertus for extraction and Qwen for discovery. Pair it with `--geometry-input`
for a completely offline graph check. `--no-constrained-decoding` applies to
extraction, discovery, and final planning; validators remain active.

`--artifact-dir` changes the intermediate-file directory (default: the output
directory). `--debug-visualizations` also saves stage plots; DINO/SAM plotting
requires matplotlib. Dataset/output paths resolve against the working directory.
Subset runs score as missing answers for omitted instructions; preserve complete
runs in separate output directories.

## Graph construction

IDs and full labels are preserved. Explicit label words determine kind and
color; conflicting colors yield null. Body parts are human, trays are containers,
plates/saucers/coasters are surfaces, and other objects are movable. The builder
does not split merged detections, resolve conflicting identities, invent missing
destinations, or consult ground truth or instruction categories.

SAM masks are reloaded and checked for size, binary foreground, saved area,
centroid, bounds, and safe-pixel membership. Boxes use mask bounds converted
from exclusive SAM endpoints to inclusive planner endpoints. `confidence`
carries the DINO score for compatibility, not a calibrated probability.

Projected `inside` edges become membership only for a nonhuman source and a
container target. If multiple containers qualify, the smallest mask wins.
Containment never creates support.

By default, selected geometry `on` hypotheses become first-pass `on_top_of`
estimates for movable objects on nonhuman objects. `supports` is derived in
code. These are **not physical confirmation**: single-view masks can be wrong
about stacking. `--no-support-candidates` omits them for comparison. Cyclic or
conflicting relationships are omitted with reasons in debug output.

Human-holding and occlusion flags are copied only from explicit upstream boolean
observations, defaulting to false when absent. Current geometry does not detect
either condition. Overlap or touching alone never creates a holding claim.

Every graph passes the existing strict validator: IDs, required fields, image
bounds, box/point consistency, references, support inverses, and relationship
cycles. This establishes structural consistency, not correct perception.
Failed builds save `<scene>.failure.json` and stop before planning.

Apertus receives the clean graph plus mask-centroid ordering, pairwise centroid
distances, and image-plane directional/overlap/touching edges. Review flags,
candidate scores, and rejected relationships stay outside the prompt. Centroids
determine spatial comparisons; safe pixels determine submission targets. The
Qwen planner retains its original prompt and pick-point-derived geometry.

## Artifacts and failures

```text
outputs/grounded_pipeline/
  object_extractions.json
  scene_discovery.json
  grounding_detections.json
  grounding_detections_clean.json
  grounding_cleanup_report.json
  segmentation_results.json
  segmentation/scene_01/red_cube_1.png
  geometry_results.json
  geometry_debug.json
  perception_config.json
  results.json
  results.raw.jsonl
  scene_graphs/
    scene_01.json
    scene_01.spatial.json
    scene_01.debug.json
    scene_01.overlay.jpg
    ...
```

Intermediate outputs are checkpointed after each instruction or scene. Upstream
errors save `perception.failure.json` with the failing stage and scene, and stop
the run. All selected graphs are checked before final planning. `--build-only`
does not write or overwrite a submission. Planning reuses the existing Apertus
client, constrained decoding, retry, and plan validation. Raw logs include the
prompt, replies, and errors. Invalid plans after retry become fail-safe refusals;
inspect these separately from model refusals. Connection/HTTP failures stop the
run rather than refusing every instruction. Successful results are checkpointed
after each instruction.

```sh
uv run python -m unittest discover -s tests -p 'test_geometry_scene_graph.py'
uv run python -m unittest discover -s tests -p 'test_grounded_pipeline.py'
```

Tests exercise the full stage ordering and handoffs with mocked model inference
and server replies, real detection cleanup/mask geometry, graph validation,
submission pixels, validation retries, and failures without weights or a GPU.
