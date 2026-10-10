# Qwen perception and Apertus planning

The Apertus extraction → Qwen discovery → DINO → SAM → geometry path has its own
`run_grounded.py` entry point (`run_geometry.py` is a compatibility alias).
See [grounded perception and planning](GEOMETRY_PIPELINE.md) for the full pipeline,
saved-geometry replay, separate results, and evaluation. The Qwen pipeline below
is unchanged.

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

The scene graph format is defined in `schema/scene_graph.schema.json`. Model output is also checked in Python for image bounds, unique IDs, valid relationship references, and pick points inside their bounding boxes.

The perception-output contract is derived from that schema by excluding `supports`. Qwen predicts `inside`, `on_top_of`, `held_by_human`, and `partially_occluded` only. Both its prompt and token filter use this reduced contract. Raw observations are validated, including containment and support cycle checks, before code builds inverse `supports` lists from `on_top_of`. Containment alone does not create support edges. The completed graph is validated again before caching or planning, so the final scene-graph format and Apertus inputs remain unchanged. Failed raw replies contain no derived support lists.

Qwen uses LM Format Enforcer through Transformers' `prefix_allowed_tokens_fn` to constrain generated JSON structure, required fields, types, and enums. The tokenizer data is cached once, while each generation attempt gets a fresh schema parser. There is no automatic unconstrained fallback. Missing decoding dependencies stop generation with an error.

Constraints do not guarantee correct perception, valid object references, or every JSON Schema keyword. The validator remains responsible for bounds and relationship consistency. A token limit can still truncate JSON; Qwen rejects incomplete responses rather than extracting a nested fragment. Validation feedback reports exact missing and unexpected fields, and failed attempts are saved in `outputs/scene_graphs/<scene>.failure.json` (or the configured graph directory) before the pipeline stops.

Other Transformers-based generative VLMs can reuse `SchemaConstrainedDecoder` with their tokenizer and pass its token filter to `generate`. Detection/segmentation backends such as Grounding DINO + SAM construct graphs in code and use validation; they do not perform JSON token decoding.

Apertus's intermediate output contract is defined in `schema/plan.schema.json` and loaded for constrained decoding. The final submission contract is defined in `schema/result.schema.json`.

## Code organization

All modules live directly under `src/`:

| Module | Responsibility |
| --- | --- |
| `prompting.py` | Perception and planning prompts, spatial calculations, image messages, and validation feedback |
| `schemas.py` | Loading JSON contracts and deriving the reduced perception schema |
| `validation.py` | Pixel, scene-graph, and plan validation |
| `scene_graph.py` | Object lookup and deterministic inverse support derivation |
| `geometry.py` | Deterministic SAM-mask geometry and reviewable support hypotheses ([usage and thresholds](GEOMETRY.md)) |
| `results.py` | Converting plans to submission entries and writing result files |
| `json_tools.py` | Extracting JSON objects from model responses |
| `visualization.py` | Scene inspection overlays |
| `perception.py` | Shared abstract interface for scene-graph generation |
| `decoding.py` | Reusable JSON-schema token filtering for Transformers-based VLMs |
| `qwen.py` | Accelerator selection, Qwen model loading, inference, and validated perception |
| `apertus.py` | Apertus requests and validated reasoning |
| `pipeline.py` | Executing perception and planning across scenes |

The model classes call the shared helpers directly. `run.py` defines the command-line arguments, connects the models, and runs the pipeline. Helpers used in only one place stay with the code that uses them.

Perception backends inherit from `PerceptionModel` and implement `generate_scene_graph(image_path, width, height)`. Each backend handles its own loading and inference, returning a validated graph in the shared schema and a list of diagnostic replies (empty for backends without text responses). Qwen implements this interface today. A future Grounding DINO + SAM backend must convert detections and masks into the same complete scene graph, including object properties and relations, before returning it. The standalone `python -m src.geometry` stage now produces mask-based image relations and separately marked support hypotheses from `segmentation_results.json`; see [the geometry contract](GEOMETRY.md). Its intermediate output still needs semantic properties and support confirmation before conversion into the planner schema. Select the backend when constructing the pipeline in `run.py`; Apertus and the pipeline do not need backend-specific logic.
