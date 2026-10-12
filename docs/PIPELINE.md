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
3. Qwen discovers visible objects and verifies its list in a second pass.
4. Grounding DINO detects those verified categories; deterministic cleanup merges duplicates.
5. SAM segments cleaned boxes and selects foreground safe pixels.
6. Geometry calculates spatial relations and selected support hypotheses.
7. The deterministic builder constructs and strictly validates the planner graph.
8. Apertus receives that graph and each instruction to produce the submission.

`run_geometry.py` is a compatibility alias for `run_grounded.py`. The launcher
supplies model/server/device settings from `containers/deployment.toml`; direct
CLI defaults also use `swiss-ai/Apertus-v1.5-70B` and
`Qwen/Qwen3-VL-32B-Instruct`.
Use the launcher for the documented Clariden configuration. All stages share the
root dependencies and one container; SAM is not separately installed.

Apertus selects object IDs and actions. Code copies the first picked object's
SAM-derived safe pixel into the submission's `target_pixel`; Apertus does not
choose pixel coordinates. Pixel-to-robot conversion is outside this evaluation
pipeline. Challenge ground truth is used only by the evaluator.

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
does not automatically add it to the default detector vocabulary.

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

Every graph passes the existing strict validator: IDs, required fields, image
bounds, box/point consistency, references, support inverses, and relationship
cycles. This establishes structural consistency, not correct perception.
Failed builds save `<scene>.failure.json` and stop before planning.

Apertus receives the clean graph plus mask-centroid ordering, pairwise centroid
distances, and image-plane directional/overlap/touching edges. Review flags,
candidate scores, and rejected relationships stay outside the prompt. Centroids
determine spatial comparisons; safe pixels determine submission targets. The
earlier Qwen graph workflow retains its original planning prompt and
pick-point-derived geometry.

## Checkpoints and failures

For the output file tree and evaluation command, see
[Run the full pipeline](CLARIDEN.md#run-the-full-pipeline).
Standalone perception scripts default to `outputs/grounded_pipeline/`; the
Clariden full-run command overrides this with a unique scratch run directory.

Intermediate outputs are checkpointed after each instruction or scene. Upstream
errors save `perception.failure.json` with the failing stage and scene, and stop
the run. All selected graphs are checked before final planning. `--build-only`
does not write or overwrite a submission. Planning reuses the existing Apertus
client, constrained decoding, retry, and plan validation. Raw logs include the
prompt, replies, and errors. Invalid plans after retry become fail-safe refusals;
inspect these separately from model refusals. Connection/HTTP failures stop the
run rather than refusing every instruction. Successful results are checkpointed
after each instruction.

## Earlier Qwen graph workflow

`run.py` is a separate workflow: Qwen2.5-VL generates a scene graph directly,
then Apertus plans from it. It does not run DINO, SAM or mask geometry, and its
`--reuse-scene-graphs` option omits the grounded pipeline's separate spatial
prompt context. For evaluating grounded perception, use `run_grounded.py`.

`run.py` retains its defaults (Qwen2.5-VL-7B-Instruct and Apertus 8B). Its
`--reuse-scene-graphs` skips Qwen inference only when a cached graph exists and
still validates that graph against the image dimensions. Default outputs are
`outputs/results.json`, `outputs/results.raw.jsonl` and `outputs/scene_graphs/`.

Qwen's perception contract uses `schema/scene_graph.schema.json` with `supports`
excluded. It predicts `inside`, `on_top_of`, holding and occlusion flags. Code
validates raw observations, derives inverse support lists from `on_top_of`,
then validates the complete graph before caching or planning. Containment alone
does not create support.

`src.decoding.SchemaConstrainedDecoder` uses LM Format Enforcer through
Transformers' `prefix_allowed_tokens_fn` to constrain JSON structure. Tokenizer
data is cached, with a fresh schema parser per generation attempt. Missing
dependencies stop generation; there is no automatic unconstrained fallback.
Validators still check image bounds, references and relation cycles. Truncated
JSON is rejected, and failed perception attempts save `<scene>.failure.json`
before stopping. DINO/SAM construct graphs in code rather than generating JSON.

Apertus planning uses `schema/plan.schema.json`; the final submission uses
`schema/result.schema.json`.

## Technical references

- [Geometry](GEOMETRY.md): mask measurements, relation thresholds and output contract.
- [Environment](ENVIRONMENT.md): dependency groups, development setup and import checks.
- [Data](DATA.md): dataset layout and coordinate conventions.
- [Evaluation](../EVALUATION.md): scoring rules and submission format.

CPU tests in `tests/test_grounded_pipeline.py` and
`tests/test_geometry_scene_graph.py` cover stage handoffs, mask geometry, graph
validation, target pixels and failure handling with mocked model inference.
The Clariden guide records which GPU deployment checks have actually passed.
