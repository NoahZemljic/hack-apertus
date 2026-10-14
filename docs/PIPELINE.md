# Pipeline architecture

To reproduce the complete grounded pipeline on Clariden, follow
[Run the full pipeline](CLARIDEN.md#run-the-full-pipeline). That is the central
run guide: container build/cache preparation, GPU allocation, startup, unique
scratch output directories, evaluation and planning replay are documented there.
This page explains the processing contracts.

## Grounded pipeline

The Clariden launcher executes `run_grounded.py` with Apertus 1.5 70B and
Qwen3-VL-32B-Instruct. Its stages are:

1. Apertus extracts object descriptions from each instruction.
2. Descriptions are combined per scene and passed to Qwen with the image.
3. The box detector is selected with `--box-detector dino|qwen` (default `dino`).
   DINO mode uses Qwen category discovery and verification, then grounds each
   verified category with its own DINO caption. Qwen mode uses one joint visual
   request to discover individual objects and their boxes, including categories
   outside the hints. Repeated instances receive separate boxes and IDs.
4. Both modes use the same deterministic box cleanup and duplicate merging.
5. SAM segments cleaned boxes and selects foreground safe pixels.
6. Geometry calculates spatial relations and selected support hypotheses.
7. The deterministic builder constructs and strictly validates the planner graph.
8. Apertus makes one planning decision per instruction; code parses its JSON and writes the submission.

In DINO mode Qwen makes two category-discovery calls per scene. In Qwen box mode
the category-only calls and DINO are skipped: a valid box response completes
the visual task immediately. `--qwen-box-retries` defaults to one retry for
malformed JSON, schema or coordinate format; it adds no inventory review,
stacking observer or planner check. Qwen uses the existing local HTTP server.
No second local Qwen model is loaded. The final Apertus prompt uses the validated
graph, mask-derived spatial context and `prompts/apertus_system.txt`.

Qwen outputs normalized half-open `[x_min,y_min,x_max,y_max]` boxes on a 0–1000
scale; the adapter validates bounds and converts once to source-image pixels.
SAM receives those pixel boxes and supplies the same masks, centroids and picking
points as in DINO mode. Qwen is not asked for target points or relations. It
supplies no detector confidence: `score` and graph `confidence` are `null`, while
SAM quality is recorded separately as `sam_score`. Legacy numeric DINO scores
remain supported.

The box-detector prompt in `prompts/qwen_boxes_system.txt` is adapted from
`feature/qwen-only`: it preserves the region-by-region scan, repeated and
partially occluded instances, low-contrast components, and tight visible boxes.
Qwen emits snake_case labels (identified fruits use just their fruit name);
the adapter normalizes labels to the shared space-separated graph format.
The bounded black placement mat keeps the benchmark name `black_tray`.
Generation uses temperature zero, a 4096-token output budget and
`enable_thinking=false`, matching the branch's non-sampling, non-thinking setup.
The branch's target-point output, separate inventory review and stacking calls
are omitted so SAM still supplies picking points in this single-pass experiment.

`run_grounded.py` is the only pipeline entry point. The launcher
supplies model/server/device settings from `containers/deployment.toml`; direct
CLI defaults also use `swiss-ai/Apertus-v1.5-70B` and
`Qwen/Qwen3-VL-32B-Instruct`.
Use the launcher for the documented Clariden configuration. All stages share the
root dependencies and one container; SAM is not separately installed.

Apertus selects object IDs and actions. Code copies the first picked object's
SAM-derived safe pixel into the submission's `target_pixel`; Apertus does not
choose pixel coordinates. Pixel-to-robot conversion is outside this evaluation
pipeline. Challenge ground truth is used only by the evaluator.

## Planning prompt and saved input

The planning policy lives in `prompts/apertus_system.txt` and is loaded by
`src/prompting.py`. It asks Apertus to resolve the instruction's references,
choose a decision, and produce the complete requested plan. The model handles
explicitly unsafe operations in that same decision. Field guidance is generated from
the supplied graph, so optional or unavailable fields are not described as
observations. Pixel geometry alone cannot establish physical support or grasp
width. Missing measurements and low detection confidence alone do not justify
refusing ordinary abstract pick/place requests. Geometry support links are
estimates used to interpret stacking and plan access.

Apertus receives every instruction and chooses plan, ask_user or refuse. There
is one planning pass with the original one-retry allowance for malformed JSON
or schema structure. A valid response returns immediately. The original flat
plan schema checks fields and types; code does not interpret the instruction,
bind a source, simulate actions, or revise the model's semantic decision. The
format-repair message only asks for corrected JSON. Repeated format failures
still become logged fail-safe refusals in the existing runner. An HTTP 400 from
schema decoding retains the existing unconstrained transport fallback.

The prompt ends with the exact instruction and supplies computed distance
ordering based on mask centroids. It defines ordinary abstract tray placement
without requiring unrelated certification of tray capacity or stability.

`run_manifest.json` records source paths and hashes, settings and the planning
policy hash. `planning_policy.txt` saves that policy. Each raw entry also contains
the exact planning and repair requests, including validation failures.

Runs save the exact planner graph in
`<output-directory>/scene_graphs/scene_XX.json`, with the additional spatial
input in `scene_XX.spatial.json`. The full per-instruction prompt and responses
are in `results.raw.jsonl`. There is one planner format, defined in
`schema/scene_graph.schema.json`: each object has a label, kind, color,
inclusive `bbox_2d`, SAM-derived `pick_point_2d`, confidence and relation fields.
The spatial file supplies mask-centroid comparisons, and `scene_XX.debug.json`
stores diagnostics; neither is another planner graph.

## Pipeline options and replay behavior

For runnable replay commands, see
[Plan and evaluate from saved grounding artifacts](CLARIDEN.md#plan-and-evaluate-from-saved-grounding-artifacts).
Use `python run_grounded.py --help` for the complete CLI. Model/server/device
settings belong in the deployment config; other pipeline flags follow the
launcher's `--` separator.

The default starts from instructions and images each time. `--geometry-input`
(also `--input`) explicitly skips all upstream perception and reuses saved
geometry and masks. Relative paths in that file resolve against its directory,
or `--base-dir`. Fresh runs write absolute image/mask paths so their generated
geometry can be replayed from another working directory.

Only Qwen's verified categories reach DINO by default. `--query-source
grounding_queries` combines them with Apertus-extracted descriptions as an
explicit experiment. An instruction mentioning an absent object therefore
does not automatically add it to the default detector vocabulary. This option
applies only to DINO; the Qwen box detector uses hints as attention guidance and
discovers every visibly relevant instance in the full image.

DINO runs one forward pass per category using the same loaded model and image.
Each returned box receives exactly one `label`: the normalized category used
for that pass. This prevents text spans from adjacent categories being fused
into labels such as `red gear yellow gear`. Returned backend text is saved
separately as `text_label`, alongside the requested `query`, for diagnosis.
Cleanup merges boxes with IoU at least 0.90 across all categories and keeps the
highest-scoring box's single label; alternative names stay in diagnostic
metadata rather than the planner graph. This guarantees one label per cleaned
detection, but boxes below that overlap threshold may still represent the same
physical object. A query returning no detections stays absent. Discovery and
verification share naming rules: observed color plus singular physical category,
without redundant bare categories when all visible instances have colored names.
Unknown colors and human body parts do not require invented color prefixes.
Both prompts scan small components, occlusions and placement-region interiors.
They follow the challenge's workspace convention: the raised grey container is
the `grey tray`, and the flat black placement mat is the `black tray`, even without
raised walls. These regions are included when visible, rather than inserted
unconditionally from instruction hints.

`--limit` caps final planning and skips perception for later scenes. Every
instruction in each included scene is still extracted to build a shared scene
inventory. `--build-only` skips final planning; fresh perception still calls
Apertus for extraction and Qwen for discovery. Pair it with `--geometry-input`
for a graph check with no model calls. `--no-constrained-decoding` applies to
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

Human-contact safety depends on any explicit upstream flags and the instruction
text; absent flags do not confirm the absence of human contact. Geometry replay
uses saved geometry and masks only. `--geometry-input --build-only` needs no
models.

Every graph passes the existing strict validator: IDs, required fields, image
bounds, box/point consistency, references, support inverses, and relationship
cycles. This establishes structural consistency, not correct perception.
Failed builds save `<scene>.failure.json` and stop before planning.

Apertus receives the clean graph plus mask-centroid ordering, pairwise centroid
distances, and image-plane directional/overlap/touching edges. Review flags,
candidate scores, and rejected relationships stay outside the prompt. Centroids
determine spatial comparisons; safe pixels determine submission targets.

## Checkpoints and failures

For the output file tree and evaluation command, see
[Run the full pipeline](CLARIDEN.md#run-the-full-pipeline).
Standalone perception scripts default to `outputs/grounded_pipeline/`; the
Clariden full-run command overrides this with a unique scratch run directory.

Intermediate outputs are checkpointed after each instruction or scene. Upstream
errors save `perception.failure.json` with the failing stage and scene, and stop
the run. All selected graphs are checked before final planning. `--build-only`
does not write or overwrite a submission. Planning reuses the existing Apertus
client, constrained decoding, one format retry, and JSON schema validation.
Raw logs include the prompt, replies, and errors. Invalid plans after retry become fail-safe refusals;
inspect these separately from model refusals. Connection/HTTP failures stop the
run rather than refusing every instruction. Successful results are checkpointed
after each instruction.

## Schemas and benchmark

In DINO mode Qwen discovers and verifies category names using
`schema/scene_discovery.schema.json`, then DINO supplies detections. In Qwen box
mode Qwen supplies instances using `schema/qwen_boxes.schema.json`. Both use SAM
for masks, and Python builds `schema/scene_graph.schema.json`. Apertus extraction uses
`schema/object_extraction.schema.json`, planning uses `schema/plan.schema.json`,
and the final submission uses `schema/result.schema.json`.

`baseline/baseline.py` is the challenge's supplied Apertus benchmark. It is kept
for score comparison and does not participate in this pipeline. For benchmark
instructions, see [the comparison guide](../instructions.md).

## Technical references

- [Geometry](GEOMETRY.md): mask measurements, relation thresholds and output contract.
- [Environment](ENVIRONMENT.md): dependency groups, development setup and import checks.
- [Data](DATA.md): dataset layout and coordinate conventions.
- [Evaluation](../EVALUATION.md): scoring rules and submission format.

CPU tests in `tests/test_grounded_pipeline.py` and
`tests/test_geometry_scene_graph.py` cover stage handoffs, mask geometry, graph
validation, target pixels and failure handling with mocked model inference.
The Clariden guide records which GPU deployment checks have actually passed.
