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
| `vision` extra | Transformers for DINO/SAM/Qwen, Qwen image utilities, constrained decoding, headless Matplotlib plots, and the OpenAI-compatible client used by standalone scene discovery. |
| `accelerator` extra | Torch and Torchvision, **only when the container does not already provide its matching builds**. |

The upstream development lock resolves Transformers 5.18.0 within the root
`>=5.10.4,<6` bound. The unified image keeps Swiss AI's compatible
`5.14.0.dev0` fork; its installed package manifest is separate from `uv.lock`. SAM uses Transformers' built-in SAM 2
implementation, including `Sam2Model` and `Sam2Processor`; it needs no separate
SAM repository or CUDA extension. See the [SAM 2 API](https://huggingface.co/docs/transformers/model_doc/sam2).

## Install inside the GPU container

For the unified image follow [CLARIDEN.md](CLARIDEN.md); all dependencies are
baked in. From the login node, after building/exporting the SquashFS image:

```sh
cd "$HOME/hack-apertus"
srun -A <project_account> --partition=debug --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --cpus-per-task=64 --time=00:15:00 \
  --environment=./containers/clariden.edf.toml --pty bash
```

Inside the container:

```sh
cd /opt/hack-apertus
python --version
python -m pip check
python -m src.check_environment --require-cuda --check-vllm
```

Do not run `uv sync` or install upstream vLLM/Transformers into this image: those
would replace the tested Apertus forks. The EDF disables the entrypoint, so the
shell command starts no servers. Start the automatic launcher or manual servers
as described in the Clariden guide. Run `srun` on the host/login node; use `hf`,
`vllm` and pipeline scripts inside the container.

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
Torchvision NMS, DINO/SAM/Qwen class imports, SAM box processing and mask
restoration, constrained-decoder utilities, headless PNG plotting, and project
imports. It downloads no weights and makes no model inference requests.
Missing packages or incompatible binaries produce named failures and a nonzero
exit status. A CPU result does not count as verified GPU execution.

The decoder uses LM Format Enforcer's public core API through `src.decoding`.
Its older Transformers adapter imports a tokenizer symbol removed in
Transformers 5; the project wrapper avoids that import and keeps schema token
filtering active. The preflight exercises a real tokenizer and allowed-token
grammar, not just package metadata.

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

## Cache weights, then run offline

For the unified deployment, accept Apertus's gated access conditions with the
same Hugging Face account used by the CLI. Then, inside the container:

```sh
HF_HUB_OFFLINE=0 hf auth login   # if needed
bash containers/start_pipeline.sh --prepare-cache
```

This prepares Apertus 1.5 70B, Qwen3-VL-32B-Instruct, Grounding DINO and SAM 2 in
the EDF's persistent `HF_HOME=$SCRATCH/hack-apertus/huggingface`. It downloads no
weights during the image build and starts no servers during cache preparation.
If the Xet downloader fails, use
`HF_HUB_DISABLE_XET=1 bash containers/start_pipeline.sh --prepare-cache`.
For Apertus-only geometry replay, see the single-model download and startup
commands in [CLARIDEN.md](CLARIDEN.md).

Normal startup is offline and uses local snapshots and localhost HTTP servers.
Custom development model IDs need their own complete cached snapshots, including
processor/tokenizer files. Preparation and serving must share the same cache.

For an existing geometry artifact, no servers or weights are needed:

```sh
python run_grounded.py --geometry-input outputs/grounded_pipeline/geometry_results.json --build-only
```

Check all stages without model weights or server connections:

```sh
python -m unittest discover -s tests
```

SAM standalone usage is `python -m src.masking.sam`. Its default input is
`outputs/grounded_pipeline/grounding_detections_clean.json`; it saves one best nonempty mask per detection,
unchanged detection metadata, centroid/area/bounds, and a distance-transform
interior safe pixel. Masks, `segmentation_results.json` and overlays all go under
`outputs/grounded_pipeline/`, in its `segmentation/` and `segmentation_debug/`
subdirectories where appropriate.
These stages all use this shared environment. See [the grounded pipeline](GEOMETRY_PIPELINE.md)
for orchestration, saved-geometry replay, output paths, and failure handling.

## What still requires a cluster run

Passing the local checks/tests does not certify a particular image's GPU driver,
vLLM kernels, model checkpoint loading, or generated answers. Run the CUDA,
engine, and local-server checks inside the actual EDF allocation, followed by
a complete pipeline run. The ARM64 unified image has been built/imported and
Apertus 70B planning replay has run on two GH200 GPUs with the fusion workaround.
The remaining concurrent Qwen/DINO/SAM checks and complete-run timing are still
pending. The unified build pins an inspected ARM64 digest; see [CLARIDEN.md](CLARIDEN.md)
for its compatibility evidence and exact GPU acceptance commands.

The under-15-minute build/run target needs measurement on that allocation,
including whatever image preparation, dependency installation, weight loading,
kernel compilation, and inference the jury timer includes. These dependencies
and tests make installation reproducible; they do not establish that timing.
