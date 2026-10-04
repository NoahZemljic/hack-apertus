# Qwen perception and Apertus planning

The pipeline separates visual observations from robot decisions:

1. Qwen2.5-VL reads each scene once and produces a validated 2D scene graph
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

The scene graph format is defined in `schema/scene_graph.schema.json`. `jsonschema` checks its structure directly, so the runtime validator does not maintain a separate list of required scene fields. Python checks cross-object consistency and image-dependent geometry as well. Independent structural and semantic errors are collected into one correction request instead of stopping at the first invalid box.

The perception-output contract is derived from that schema by excluding `supports`. Qwen predicts `inside`, `on_top_of`, `held_by_human`, and `partially_occluded` only. Both its prompt and token filter use the same reduced contract with the actual image dimensions pinned as enums. Raw observations are validated before code builds inverse `supports` lists from `on_top_of`. The pipeline validates complete graphs from every perception backend before caching or planning. Failed raw replies contain no derived support lists.

Qwen uses LM Format Enforcer through Transformers' `prefix_allowed_tokens_fn` to constrain generated JSON structure, required fields, types, and enums. The tokenizer data is cached once, while each generation attempt gets a fresh schema parser. There is no automatic unconstrained fallback. Missing decoding dependencies stop generation with an error.

Constraints do not guarantee correct perception, valid object references, or every JSON Schema keyword. The validator remains responsible for bounds and relationship consistency. A token limit can still truncate JSON; Qwen rejects incomplete responses rather than extracting a nested fragment. Duplicate JSON keys and non-finite numbers are also rejected. Validation feedback includes field paths, observed values, and correction guidance. Failed attempts are saved in `outputs/scene_graphs/<scene>.failure.json` (or the configured graph directory) before the pipeline stops. No validated graph or overlay is written for that failed attempt; use a fresh output directory to avoid confusing artifacts from earlier runs with the current result.

Other Transformers-based generative VLMs can reuse `SchemaConstrainedDecoder` with their tokenizer and pass its token filter to `generate`. Detection/segmentation backends such as Grounding DINO + SAM construct graphs in code and use validation; they do not perform JSON token decoding.

Apertus's intermediate output contract is defined in `schema/plan.schema.json` and loaded for constrained decoding. The final submission contract is defined in `schema/result.schema.json`.

## Scene semantics and limits

This is a task-specific overhead tabletop model, not a general representation of arbitrary 3D arrangements.

| Field | Meaning and checks |
| --- | --- |
| `bbox_2d` | Half-open `[x1, y1, x2, y2]`: `0 <= x1 < x2 <= width`, `0 <= y1 < y2 <= height`. Right/bottom edges may equal image dimensions |
| `pick_point_2d` | Integer pixel `[u, v]` inside the half-open box. Unlike box edges, `u < width` and `v < height` |
| `inside` | Immediate enclosing receptacle, directed from contents to receptacle. It does not by itself assert supporting contact |
| `on_top_of` | Direct supporting object below this object, never the reverse direction. It does not assert enclosure |
| `supports` | Computed inverse of `on_top_of`, never independently predicted or inferred from `inside` |
| `kind` | Manipulation role: mugs/cups/bowls and other graspable objects are `movable`, trays `container`, plates/saucers `surface`, hands `human` |

An object can be both inside a tray and resting on its base. A movable mug can contain another object; manipulation role and receptacle capability are not the same thing. The validator recognizes trays/other objects classified as containers and the movable receptacle labels `mug`, `cup`, and `bowl`. Adding a new movable receptacle category requires updating that explicit policy. Solid gears and cubes, plates, and human hands are not enclosure targets in this model.

The background table and floor are excluded from the object list, with `on_top_of: null` for background support. Actual black trays must remain detections. Null means no represented relation was established; it is not proof that the object is unsupported, uncontained, or safe to pick. An empty workspace can use `objects: []`, so the schema does not force a detection.

IDs must be unique, references must resolve, and neither containment nor support may form a cycle. Mixed chains, such as `A.inside=B`, `B.on_top_of=C`, `C.inside=A`, are rejected too. A conservative containment gate requires the contained object's box to fit within the receptacle's box, allowing a margin of 2% of the longer image dimension. This rejects claims such as a large tray inside a small mug, but is only a 2D plausibility heuristic: protruding objects, perspective, occlusion, or unusual shapes can require human review. Box overlap never establishes containment or support on its own.

Validation never silently reverses an edge, invents support, changes a category, clamps a pick point, or modifies a box. Qwen receives the violations and original image for a bounded retry. The graph is rejected if it still fails.

Passing these checks means the graph satisfies this contract, not that the image was interpreted correctly. In particular, a mistaken gear label, a missed object, a pick point over a hole, false contact, or a falsely negative human-holding flag can still be internally consistent. Self-reported confidence is not calibrated safety evidence. Inspect overlays and compare detections with ground truth during evaluation; depth/contact/grasp checks and an independent safety controller are still needed before physical execution.

## Updating an existing Clariden environment

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
| `apertus.py` | Apertus requests and validated reasoning |
| `pipeline.py` | Executing perception and planning across scenes |

The model classes call the shared helpers directly. `run.py` defines the command-line arguments, connects the models, and runs the pipeline. Helpers used in only one place stay with the code that uses them.

Perception backends inherit from `PerceptionModel` and implement `generate_scene_graph(image_path, width, height)`. Each backend handles its own loading and inference, returning a validated graph in the shared schema and a list of diagnostic replies (empty for backends without text responses). Qwen implements this interface today. A future Grounding DINO + SAM backend must convert detections and masks into the same complete scene graph, including object properties and relations, before returning it. Select the backend when constructing the pipeline in `run.py`; Apertus and the pipeline do not need backend-specific logic.
