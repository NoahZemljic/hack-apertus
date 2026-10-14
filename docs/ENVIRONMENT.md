# Shared environment for the complete pipeline

All application dependencies are declared in the root `pyproject.toml` and
resolved in `uv.lock` for upstream development. Container images provide the
inference engine and CUDA runtime. The complete ARM64 single-container deployment
is in [the Clariden build and launch guide](CLARIDEN.md): Apertus 70B on two GPUs,
Qwen3-VL-32B on one, and DINO/SAM on one. It preserves Swiss AI's custom engine
and records a separate container package manifest.

| Dependency group | What it provides |
| --- | --- |
| Base project | NumPy, headless OpenCV, Pillow, and dependency-version checking. Geometry, mask I/O, and saved-graph building need no model inference. |
| `vision` extra | Transformers for DINO/SAM, headless Matplotlib plots, and the OpenAI-compatible client used by standalone scene discovery. Apertus and Qwen generation run through local vLLM servers. |
| `accelerator` extra | Torch and Torchvision, **only when the container does not already provide its matching builds**. |

The upstream development lock resolves Transformers 5.18.0 within the root
`>=5.10.4,<6` bound. The unified image keeps Swiss AI's compatible
`5.14.0.dev0` fork; its installed package manifest is separate from `uv.lock`. SAM uses Transformers' built-in SAM 2
implementation, including `Sam2Model` and `Sam2Processor`; it needs no separate
SAM repository or CUDA extension. See the [SAM 2 API](https://huggingface.co/docs/transformers/model_doc/sam2).

## Development installation

The unified Clariden image already includes the dependencies. Use
[the Clariden guide](CLARIDEN.md#run-the-full-pipeline) for building, opening the
container and running the pipeline. Do not run `uv sync` or install upstream
vLLM/Transformers there; doing so replaces the Apertus forks.

For a separately provided generic development container, Python 3.11–3.13 and a
matching CUDA Torch/Torchvision pair can be combined with the upstream lock:

```sh
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'uv==0.13.0'
uv sync --active --locked --inexact --extra vision
python -m src.check_environment --require-cuda
python -m pip check
```

This optional development recipe is not startup work for the unified image.
`--system-site-packages` preserves the supplied Torch pair; do not select
`--extra accelerator` when that pair is already provided.

For a fresh machine without a provided Torch build, install the `accelerator`
extra as well. Select the correct CUDA/CPU wheels for that machine using
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
Torch and Torchvision must be a matching pair. One reproducible **CPU-only**
smoke-test setup is:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'uv==0.13.0'
python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cpu
uv export --frozen --extra vision --no-hashes --no-emit-project --output-file /tmp/hack-apertus-vision.lock.txt
python -m pip install -r /tmp/hack-apertus-vision.lock.txt
python -m src.check_environment
```

The CPU wheel pair satisfies the centrally declared `accelerator` bounds.
The exported `vision` requirements deliberately exclude Torch, so neither
installation path replaces the supplied pair. CPU smoke checks validate the
software boundary; the model pipeline should use GPUs on Clariden.

## Check before loading models

```sh
python -m src.check_environment
python -m src.check_environment --require-cuda --check-vllm
python -m src.check_environment --require-cuda --check-servers
python -m src.check_environment --profile base
python -m src.check_environment --json
```

The default vision check validates versions against `pyproject.toml`, actual
NumPy/OpenCV/Pillow operations, Torch-to-NumPy interoperability, compiled
Torchvision NMS, DINO/Qwen3-VL/SAM class imports, SAM box processing and mask
restoration, headless PNG plotting, and project
imports. It downloads no weights and makes no model inference requests.
Missing packages or incompatible binaries produce named failures and a nonzero
exit status. A CPU result does not count as verified GPU execution.

Extraction, category discovery and final planning send JSON-schema
`response_format` requests to the local vLLM servers. The application does not
load a Qwen tokenizer or maintain a local token-filtering decoder. JSON parsing
and structural validation still check each returned payload.

`--require-cuda` additionally tests a CUDA tensor operation and Torchvision's
CUDA operator. `--check-vllm` imports the container-provided engine. These checks
must run inside the intended compute image; vLLM is not installed by the
application extras because the EDF image supplies it.

`--check-servers` queries `/v1/models` on local Apertus (port 8000) and Qwen
(port 8001), checking the expected model names without generating anything.
Override URLs/model names with the same `--apertus-*` / `--qwen-*` options used
by `run_grounded.py`. Server inventory checks establish reachability and names,
not successful generation or throughput. Separate server processes need
separate GPU assignments or sufficient explicitly shared GPU memory.

The full grounded pipeline uses local OpenAI-compatible HTTP endpoints. These
are local model servers, not external hosted APIs. The OpenAI Python package
does not imply that data is sent to an external provider.

## Cache preparation and full runs

Use [Persistent cache preparation](CLARIDEN.md#persistent-cache-preparation)
for model downloads and [Run the full pipeline](CLARIDEN.md#run-the-full-pipeline)
for offline inference and evaluation. Preparation and serving must share the
same persistent cache. Custom development model IDs need complete cached
snapshots, including processor/tokenizer files.

To check the software without model weights or server connections:

```sh
python -m unittest discover -s tests
```

See [Pipeline architecture](PIPELINE.md) for stage contracts and replay behavior.
