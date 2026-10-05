# Qwen perception and Apertus planning

The pipeline separates visual observations from robot decisions:

1. Qwen produces a scene graph; Qwen3 additionally checks placement mats, overlapping-object contact, and close-up pick points before validation
2. The graph is cached as JSON so detections and relationships can be inspected
3. Apertus receives the graph, an instruction, and the robot safety constraints
4. Apertus chooses a plan, clarification question, or refusal
5. For a plan, ordinary code copies the first picked object's `pick_point_2d` into `target_pixel`

Pixel-to-robot conversion is deliberately outside the reasoning path. The challenge evaluator consumes full-resolution pixels. A physical robot can later pass an accepted target through `starter/scene_io.py`.

## GPU environment

Create the environment on a compute node from a CSCS PyTorch container so its CUDA-compatible PyTorch installation is available. Then install the optional model libraries without replacing the container's PyTorch build:

```sh
python -m venv --system-site-packages .venv
uv sync --active --extra vision
```

On CSCS, use a recent PyTorch container on a Clariden compute node rather than a login node. Put Hugging Face model caches on scratch or project datacache, not in the small home filesystem.

Start Apertus in a separate allocation or process:

```sh
uv tool run --from vllm vllm serve swiss-ai/Apertus-v1.5-8B --chat-template-content-format string
```

Then run the pipeline:

```sh
uv run --extra vision run.py
```

Useful development options:

```sh
uv run --extra vision run.py --limit 5
uv run --extra vision run.py --reuse-scene-graphs
uv run --extra vision run.py --attention flash_attention_2
```

`--reuse-scene-graphs` skips Qwen inference only when a graph already exists and still validates the cached graph against the current image dimensions.

## Generated artifacts

- `outputs/results.json`: challenge submission
- `outputs/results.raw.jsonl`: raw perception and planning responses plus validation errors
- `outputs/scene_graphs/scene_NN.json`: validated Qwen scene graphs
- `outputs/scene_graphs/scene_NN.overlay.jpg`: bounding-box and pick-point inspection image
- `outputs/scene_graphs/scene_NN.raw.json`: draft, refinement, and correction replies saved immediately per scene

The scene graph format is defined in `schema/scene_graph.schema.json`. `jsonschema` checks its structure directly, so the runtime validator does not maintain a separate list of required scene fields. Python checks cross-object consistency and image-dependent geometry as well. Independent structural and semantic errors are collected into one correction request instead of stopping at the first invalid box.

The perception-output contract is derived from that schema by excluding `supports`. Qwen predicts `inside`, `on_top_of`, `held_by_human`, and `partially_occluded` only. Both its prompt and token filter use the same reduced contract with the coordinate-plane dimensions pinned as enums. Raw observations are validated before code builds inverse `supports` lists from `on_top_of`. The pipeline validates complete graphs from every perception backend before caching or planning. Refinement diagnostics may contain assembled graphs with derived support lists.

Qwen uses LM Format Enforcer through Transformers' `prefix_allowed_tokens_fn` to constrain generated JSON structure, required fields, types, and enums. The tokenizer data is cached once, while each generation attempt gets a fresh schema parser. There is no automatic unconstrained fallback. Missing decoding dependencies stop generation with an error.

Constraints do not guarantee correct perception, valid object references, or every JSON Schema keyword. The validator remains responsible for bounds and relationship consistency. A token limit can still truncate JSON; Qwen rejects incomplete responses rather than extracting a nested fragment. Duplicate JSON keys and non-finite numbers are also rejected. Validation feedback includes field paths, observed values, and correction guidance. Failed attempts are saved in `outputs/scene_graphs/<scene>.failure.json` (or the configured graph directory) before the full pipeline stops. Perception-only runs continue to the next image and return a nonzero exit status if any graph fails. A failed regeneration removes any previous cached graph and overlay for that scene; a successful regeneration removes its old failure record. Use a fresh output directory for comparisons.

Other Transformers-based generative VLMs can reuse `SchemaConstrainedDecoder` with their tokenizer and pass its token filter to `generate`. Detection/segmentation backends such as Grounding DINO + SAM construct graphs in code and use validation; they do not perform JSON token decoding.

Apertus's intermediate output contract is defined in `schema/plan.schema.json` and loaded for constrained decoding. The final submission contract is defined in `schema/result.schema.json`.

## Scene semantics and limits

This is a task-specific overhead tabletop model, not a general representation of arbitrary 3D arrangements.

| Field | Meaning and checks |
| --- | --- |
| `bbox_2d` | Half-open `[x1, y1, x2, y2]`: `0 <= x1 < x2 <= width`, `0 <= y1 < y2 <= height`. Right/bottom edges may equal image dimensions |
| `pick_point_2d` | Integer pixel `[u, v]` inside the half-open box. Unlike box edges, `u < width` and `v < height` |
| `inside` | Immediate enclosing receptacle, directed from contents to receptacle. It does not by itself assert supporting contact |
| `on_top_of` | List of all detected objects directly supporting this object, with distinct IDs. `[]` means no represented support was established. It does not assert enclosure |
| `supports` | Computed inverse of `on_top_of`, never independently predicted or inferred from `inside` |
| `kind` | Manipulation role: mugs/cups/bowls and other graspable objects are `movable`, trays `container`, plates/saucers `surface`, hands `human` |

An object can be both inside a tray and resting on its base. A movable mug can contain another object; manipulation role and receptacle capability are not the same thing. The validator recognizes trays/other objects classified as containers and the movable receptacle labels `mug`, `cup`, and `bowl`. Adding a new movable receptacle category requires updating that explicit policy. Solid gears and cubes, plates, and human hands are not enclosure targets in this model.

The background table and floor are excluded from the object list, with `on_top_of: []` for background support. Bounded work mats are included as `surface` objects with label `work mat`, so they can serve as placement destinations. Challenge instructions call the black work mat the "black tray"; the planner resolves this phrase to that detected surface. Objects on a mat use `on_top_of`, not `inside`. Actual raised-wall trays of any color must remain detections. `inside: null` and `on_top_of: []` mean no corresponding represented relation was established; they are not proof that the object is unsupported, uncontained, or safe to pick. An empty workspace can use `objects: []`, so the schema does not force a detection.

For example, a cube resting directly across two gears has `on_top_of: ["gear_1", "gear_2"]`. Code includes the cube's ID in both gears' `supports` lists. Do not include an indirect support beneath a gear unless the cube also directly contacts it. `inside` remains a single immediate receptacle ID or null; multiple support does not imply multiple containment.

IDs must be unique, references must resolve, and neither containment nor support may form a cycle. Every support-list entry is checked, including branches beyond the first support. Mixed chains, such as `A.inside=B`, `B.on_top_of=[C]`, `C.inside=A`, are rejected too. A conservative containment gate requires the contained object's box to fit within the receptacle's box, allowing a margin of 2% of the longer image dimension. This rejects claims such as a large tray inside a small mug, but is only a 2D plausibility heuristic: protruding objects, perspective, occlusion, or unusual shapes can require human review. Direct support between boxes separated by more than the same 2% tolerance is also rejected in this overhead tabletop model. Box overlap never establishes containment or support on its own.

The perception prompt distinguishes industrial metal parts and brackets from ceramic plates/saucers, and treats industrial components as movable objects. It asks for object boundaries before relations, solid-material pick points away from all holes, separate partially occluded detections, and human-holding flags based on visible gripping. Correction feedback explicitly permits `inside: null` for an object outside a receptacle and removal of unsupported `on_top_of` entries. It does not contain scene-specific counts, coordinates, or required inventories.

Validation never silently reverses an edge, invents support, changes a category, clamps a pick point, or modifies a box. Qwen receives the violations, rejected candidate, and original image in a fresh request for a bounded retry, avoiding an accumulating assistant history of invalid answers. The graph is rejected if it still fails.

Passing these checks means the graph satisfies this contract, not that the image was interpreted correctly. In particular, a mistaken gear label, a missed object, a pick point over a hole, false contact, or a falsely negative human-holding flag can still be internally consistent. Self-reported confidence is not calibrated safety evidence. Inspect overlays and compare detections with ground truth during evaluation; depth/contact/grasp checks and an independent safety controller are still needed before physical execution.

## Updating an existing Clariden environment

Cached graphs using a string or null for `on_top_of` no longer match the schema. Regenerate them in a fresh scene-graph directory without `--reuse-scene-graphs`; old observations are not silently converted or treated as newly verified detections.

Install the new validation dependency in an existing Clariden environment without changing PyTorch:

```sh
.venv-clariden/bin/python -m pip install "jsonschema>=4.23,<5"
```

## Code organization

All modules live directly under `src/`:

| Module | Responsibility |
| --- | --- |
| `prompting.py` | Perception and planning prompts, spatial calculations, image messages, and validation feedback |
| `schemas.py` | Loading JSON contracts and deriving the reduced perception schema |
| `validation.py` | Pixel, scene-graph, and plan validation |
| `scene_graph.py` | Object lookup and deterministic inverse support derivation |
| `results.py` | Converting plans to submission entries and writing result files |
| `json_tools.py` | Strict perception/cache JSON parsing and extraction of planner JSON responses |
| `visualization.py` | Scene inspection overlays |
| `perception.py` | Shared abstract interface for scene-graph generation |
| `decoding.py` | Reusable JSON-schema token filtering for Transformers-based VLMs |
| `qwen.py` | Accelerator selection, Qwen model loading, inference, and validated perception |
| `grounding.py` | Conservative material-point checks for pale industrial parts |
| `apertus.py` | Apertus requests and validated reasoning |
| `pipeline.py` | Executing perception and planning across scenes |

The model classes call the shared helpers directly. `run.py` defines the command-line arguments, connects the models, and runs the pipeline. Helpers used in only one place stay with the code that uses them.

Perception backends inherit from `PerceptionModel` and implement `generate_scene_graph(image_path, width, height)`. Each backend handles its own loading and inference, returning a validated graph in the shared schema and a list of diagnostic replies (empty for backends without text responses). Qwen implements this interface today. A future Grounding DINO + SAM backend must convert detections and masks into the same complete scene graph, including object properties and relations, before returning it. Select the backend when constructing the pipeline in `run.py`; Apertus and the pipeline do not need backend-specific logic.

## Perception-only runs

Run perception independently of Apertus on a compute node:

```sh
CUDA_VISIBLE_DEVICES=0 .venv-clariden/bin/python -u run.py \
  --perception-only --qwen-model Qwen/Qwen3-VL-32B-Instruct \
  --scene-graph-dir outputs/vlm-review/scene_graphs
```

All selected scenes are attempted. `perception-report.json` is updated after every
scene; `valid` means the graph passes the contract, not that its visual semantics
are correct. Use repeated `--scene scene_04 --scene scene_06` for a focused run.
`--limit` counts planning instructions and is not accepted in perception-only mode.
No Apertus server is needed.

The default is Qwen3-VL-32B-Instruct in bfloat16. Allow roughly 70 GB of free GPU
memory for this dataset; the CSCS runs used one dedicated GH200 GPU per process.
Use `--qwen-model Qwen/Qwen3-VL-8B-Instruct` for a smaller comparison model, or
`Qwen/Qwen2.5-VL-7B-Instruct` to reproduce the original model path. The larger model
improves detection completeness, but does not guarantee correct relationships.

Generation allows 4096 output tokens and two validation retries. Configure them
with `--qwen-max-new-tokens` and `--qwen-validation-retries`. The larger budget
provides room for crowded scenes.
Short refinement responses are capped at 512 tokens, and point responses at 128
(or the configured maximum, if smaller). Inference logs include token counts and
elapsed seconds for each call.
Qwen3-VL uses its native normalized 0–1000 coordinate plane. Code converts boxes
and points into the source image's pixel coordinates; no planner ever receives
normalized coordinates. Box minima are rounded down, maxima up, and pixel points
down. Qwen2.5 retains the original absolute-pixel inference path for comparison.
The installed Transformers version must be at least 4.57.6 for Qwen3 support.

After Qwen3's full-image graph validates, focused model calls inspect:

- Missing work mats: a separate image query checks usable placement areas when
  the draft contains no mat, including empty mats that detection can overlook.
- Potential contacts: overlapping non-background objects are shown together in
  a crop. Qwen predicts the objects directly below each target, occlusion, and
  human holding. Overlap selects what to inspect; it never creates an edge.
- Pair consistency: overlapping pairs with a predicted contact or occlusion get one shared front/back and
  direct-contact decision, replacing conflicting per-object edges. This avoids
  two objects claiming to support each other. An unclear answer rejects the graph;
  a confidently wrong visual answer can still pass validation.
  Pairs with less than 10% projected overlap of the smaller box are skipped to
  avoid forcing a depth-order question about nearly disjoint boxes. This gate can
  miss small contacts; it is not a physical contact detector.
- Pick points: each object is shown in an unmarked crop. Qwen selects a point on
  visible material, with explicit instructions to avoid holes, cavities, and
  occluders. Code maps normalized crop coordinates back to source pixels.
- Pale industrial parts: a high-contrast, low-saturation material mask checks
  clearance from dark holes and other objects' boxes. If the proposed point lacks
  clearance, the deepest interior material pixel replaces it. The check abstains
  when contrast or exposed material is insufficient; it is not a general segmenter.

Labels use the canonical categories in the schema, with `object` as a fallback.
Color belongs in `color`, so a yellow gear cannot become the category `yellow sun`.
This controls terminology, but cannot establish that a detected object exists.

The final graph is validated again and inverse supports are rebuilt. These are
model observations, not independent verification or a guarantee of graspability.
Ground truth is not supplied to inference. Model replies and the assembled
graph are retained in each scene's diagnostic file.
