# Single-container Clariden deployment

The launcher runs the existing `run_grounded.py` pipeline: Apertus extraction →
Qwen discovery/verification → DINO → cleaned detections → SAM 2 → geometry → scene
graph → Apertus planning → results. All inference stays local. Normal startup forces
offline model loading and binds both servers to `127.0.0.1`.

| Allocation GPU positions | Process | Default model | TP | Port |
| --- | --- | --- | --- | --- |
| 0, 1 | Persistent vLLM | `swiss-ai/Apertus-v1.5-70B` | 2 | 8000 |
| 2 | Persistent vLLM | `Qwen/Qwen3-VL-32B-Instruct` | 1 | 8001 |
| 3 | Pipeline PyTorch | Grounding DINO + SAM 2 | — | — |

Positions refer to **Slurm's visible allocation**. The launcher maps through the
original `CUDA_VISIBLE_DEVICES` ordering, including GPU UUIDs. The perception child
sees only allocation GPU 3 and gets `--dino-device cuda:0 --sam-device cuda:0`.
Those local device names both refer to its sole visible GPU.

## Compatibility and dependency decisions

The [Apertus release](https://huggingface.co/swiss-ai/Apertus-v1.5-70B) requires Swiss
AI's modified vLLM and Transformers. The Containerfile pins the public ARM64 release
manifest inspected on 2026-10-10:

```text
ghcr.io/swiss-ai/vllm_apertus_1.5_release@sha256:8eb36e284fbefdd44c6f4dc23072eff9c06c1caed5643ce2f99ac7e0915a06ae
```

Registry metadata confirms ARM64, and the image history matches the
[official recipe](https://github.com/swiss-ai/model-launch/blob/main/images/vllm_apertus_1.5_release/Dockerfile):
Python 3.12, CUDA 13.0, Torch 2.11.0/Torchvision 0.26.0, vLLM commit
`a601a9d998ddeb488f0c17e8512874b116aa7658`, and Transformers commit
`3797303dda74844e3d1f8977ff5518bb91f818b4` (`5.14.0.dev0`). Its
[model registry](https://github.com/swiss-ai/vllm/blob/a601a9d998ddeb488f0c17e8512874b116aa7658/vllm/model_executor/models/registry.py)
includes both `Apertus1p5ForConditionalGeneration` and `Qwen3VLForConditionalGeneration`.
The build/startup checks import both actual implementations, plus DINO and SAM;
this establishes implementation availability, while successful GPU loading and
generation require actual GPU tests (see the validation status below). Qwen's [model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
also documents vLLM support.

All application dependencies remain centrally declared in root `pyproject.toml`.
SAM uses Transformers' SAM 2 integration, with no separately installed project,
requirements file, external SAM repository or custom extension. The image installer
freezes **every inherited package**, installs missing application packages within
the root bounds, and runs `pip check`. Conflicts fail instead of replacing the
engine. The OpenCV bound includes version 5 because this release supplies
`opencv-python-headless==5.0.0.93`; the CPU mask/geometry tests validate its operations.
The installed engine is `0.23.1rc1.dev1029+ga601a9d99.precompiled`.

`environment_manifest/{application.txt,engine_constraints.txt,installed.txt}` inside
the image records its dependency resolution. The ordinary `uv.lock` remains the
upstream development environment: **do not run `uv sync` or install generic
vLLM/Transformers into this image**, as that would replace the Apertus fork.

## Memory and time limits

Defaults: BF16, 8,192-token contexts, one simultaneous sequence per server, 90%
GPU memory utilization. Apertus image/audio prompts are disabled to avoid profiling
unused media; Qwen allows one image and no videos. Apertus 70B has now loaded and
served planning requests on **two Clariden GH200 GPUs**, with about **66.85 GiB of
model memory per GPU** at these settings. Disable fused all-reduce RMS with
`--compilation-config.pass_config.fuse_allreduce_rms false`; this is included in
the default deployment configuration and manual command below. The original
startup failed during CUDA graph capture without the workaround, as described
in the [official Apertus instructions](https://huggingface.co/swiss-ai/Apertus-v1.5-70B#vllm).
This validates the tested Apertus settings, not other context sizes or the full
four-GPU pipeline. The tested node exposed about 95 GiB of usable memory per GPU;
use actual runtime memory rather than the GPU product name to assess budgets.
Qwen's [weight index](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/main/model.safetensors.index.json)
reports 66,714,780,128 bytes (62.1 GiB). Startup compares optimistic weight/KV lower
bounds from **cached** metadata against actual GPU memory and rejects impossible
budgets. Workspaces, activations, multimodal replication, CUDA graphs and NCCL add
overhead: passing this check does not prove fit. The launcher does not silently
quantize models or change their GPU allocation.

Prepare the image and weight cache once; normal debug startup installs nothing
and downloads nothing. Persistent CUDA/vLLM caches also reduce repeated compilation.

## Build once without model weights

Use an ARM64 compute node and configure rootless Podman as in the
[CSCS build guide](https://docs.cscs.ch/build-install/containers/): storage on local
`/dev/shm`, not Lustre. From the login node:

```sh
cd "$HOME/hack-apertus"
srun -A <project_account> --partition=debug --nodes=1 --ntasks=1 \
  --cpus-per-task=64 --time=00:30:00 --pty bash
```

Inside the build allocation:

```sh
cd "$HOME/hack-apertus"
uname -m                       # aarch64
mkdir -p "$SCRATCH/hack-apertus/containers"
podman build --network=host --platform linux/arm64 \
  -f containers/Containerfile -t hack-apertus:pipeline-arm64 .
enroot import -x mount -o "$SCRATCH/hack-apertus/containers/pipeline-arm64.sqsh" \
  podman://localhost/hack-apertus:pipeline-arm64
sha256sum "$SCRATCH/hack-apertus/containers/pipeline-arm64.sqsh"
```

If your account's debug time limit is shorter, use a permitted longer build
allocation. Import before leaving: Podman's local storage is temporary. The image
build runs import checks and CPU tests; it never downloads weights. `.dockerignore`
excludes datasets, cached models, credentials, generated artifacts and `.git`.
Sources and schemas are baked into `/opt/hack-apertus`.

## Persistent cache preparation

`containers/clariden.edf.toml` mounts `$HOME` and `$SCRATCH` and uses the imported
SquashFS image. It sets `HF_HOME=$SCRATCH/hack-apertus/huggingface`, persistent XDG/
Matplotlib/CUDA/TorchInductor/Triton caches, and per-server vLLM caches under scratch. Preparation and
startup must use the same HF cache path. You can instead mount a project/datacache
location and update the EDF paths for longer retention; scratch is subject to
purging. Existing compatible caches need not be copied into the image.

Accept Apertus's gated Hub access conditions using your own account first. Keep
credentials out of the image, EDF and Git; enter a token interactively if needed.
Download access is needed only for preparation. From the login node:

```sh
srun -A <project_account> --partition=debug --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --cpus-per-task=64 --time=00:30:00 \
  --environment=./containers/clariden.edf.toml --pty bash
```

Inside the container:

```sh
cd /opt/hack-apertus
HF_HUB_OFFLINE=0 hf auth login  # only if needed for Apertus access
bash containers/start_pipeline.sh --prepare-cache
```

This explicit mode downloads the four configured checkpoints, checks their weight
shards, and records `HF_HOME/hack_apertus_cache_manifest.json`, including resolved
snapshot paths. Choose a longer preparation allocation if large downloads require
it. Pin the four revision fields to the snapshot commits, or use absolute snapshot
directories as model IDs, for repeatable runs. Normal startup fails clearly when
weights are missing instead of downloading them.

For planning replay, prepare only Apertus inside the container:

```sh
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 \
hf download swiss-ai/Apertus-v1.5-70B \
  --cache-dir "${HF_HUB_CACHE:-$HF_HOME/hub}"
```

The tested download needed `HF_HUB_DISABLE_XET=1` after a Xet hex-hash error.
For the complete cache preparation command, the same optional override is
`HF_HUB_DISABLE_XET=1 bash containers/start_pipeline.sh --prepare-cache`.
An access-denied error requires accepting the model page's conditions with the
same account used by `hf auth login`; logging in alone does not grant access.
The login node may not provide `hf`: use the container's tools. Downloads go to
mounted scratch and persist across allocations; retry an interrupted download
with the same cache. Normal inference continues to use offline mode.

## Exact Clariden debug-node acceptance commands

From the login node with the image and cache already prepared:

```sh
cd "$HOME/hack-apertus"
srun -A <project-account> --partition=debug --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --cpus-per-task=64 --time=00:15:00 \
  --environment=./containers/clariden.edf.toml --pty bash
```

Inside that container, run the following in order. Use fresh allocations for the
smoke, one-scene and full timed runs if they cannot share the 15-minute job:

```sh
cd /opt/hack-apertus
python --version
uname -m
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
python -m pip check
python -m src.clariden --check-engine
python -m unittest discover -s tests
bash containers/start_pipeline.sh --dry-run -- --scene scene_01 --limit 1
bash containers/start_pipeline.sh --preflight-only -- --scene scene_01 --limit 1

# Loads both servers; checks GPU/imports/cache/config and health/model inventories.
# Does not generate answers or run full perception.
time bash containers/start_pipeline.sh --smoke-test -- --scene scene_01 --limit 1

# Real extraction, image requests, DINO, cleaned-box SAM, geometry, graph, planning.
time bash containers/start_pipeline.sh -- --scene scene_01 --limit 1 \
  --debug-visualizations --out "$HACK_APERTUS_OUTPUT/one_scene/results.json"

# Full offline acceptance run including preflight/server startup/shutdown.
time bash containers/start_pipeline.sh -- \
  --out "$HACK_APERTUS_OUTPUT/full/results.json"
```

The EDF disables the image entrypoint so `--pty bash` opens a shell. Explicitly
invoke the launcher. From the host repo root, the equivalent batch job is:

```sh
sbatch -A <project_account> containers/clariden.sbatch
```

Run `srun` from the login/host environment, not from `/opt/hack-apertus` inside
the container: host Slurm plugins require host libraries. Select a longer time
only when permitted by your partition/account (90 minutes is `--time=01:30:00`).
The 15-minute default is the acceptance target, not a measured complete-run time.

## Open a shell and launch servers manually

There are two separate operations. `srun --environment=./containers/clariden.edf.toml
--pty bash` opens a shell in the container and starts no servers, because the EDF
sets `entrypoint = false`. Running `bash containers/start_pipeline.sh` **inside**
that shell launches both servers, waits for readiness, runs the complete pipeline,
and shuts them down. It does not allocate a node or create a container when run on
the login node.

Before using the EDF, its `image` path must be a readable SquashFS file:

```sh
ls -lh "$SCRATCH/hack-apertus/containers/pipeline-arm64.sqsh"
```

An Enroot/Pyxis `Invalid argument: .../pipeline-arm64.sqsh` startup error can mean
the export is missing. A successful Podman build alone does not create this file;
the `enroot import` step in the build section creates it. An invalid/truncated file
also needs a successful export before retrying `srun`.

For manual serving, open the shell using the four-GPU `srun` command above. Models
must be cached first; `bash containers/start_pipeline.sh --prepare-cache` prepares
weights without starting servers. For geometry replay only Apertus is needed:
start its block below and skip Qwen. For a full run, **inside the container**, start both
servers in the same allocation, preserving Slurm's original GPU mapping:

```sh
cd /opt/hack-apertus
IFS=, read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
test "${#allocated_gpus[@]}" -ge 4
manual_logs="$HACK_APERTUS_OUTPUT/manual"
mkdir -p "$manual_logs"
touch "$manual_logs/apertus.log" "$manual_logs/qwen.log"
apertus_path="$(python -c 'from src.clariden import resolve_snapshot; print(resolve_snapshot("swiss-ai/Apertus-v1.5-70B", "main"))')" || exit 1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT VLLM_API_KEY

CUDA_VISIBLE_DEVICES="${allocated_gpus[0]},${allocated_gpus[1]}" \
VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT/apertus" \
vllm serve "$apertus_path" \
  --served-model-name swiss-ai/Apertus-v1.5-70B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --chat-template-content-format string \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  > "$manual_logs/apertus.log" 2>&1 &
apertus_pid=$!

# Skip this Qwen block when replaying saved geometry.
qwen_path="$(python -c 'from src.clariden import resolve_snapshot; print(resolve_snapshot("Qwen/Qwen3-VL-32B-Instruct", "main"))')" || exit 1
CUDA_VISIBLE_DEVICES="${allocated_gpus[2]}" \
VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT/qwen" \
vllm serve "$qwen_path" \
  --served-model-name Qwen/Qwen3-VL-32B-Instruct \
  --host 127.0.0.1 --port 8001 --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":1,"video":0}' \
  > "$manual_logs/qwen.log" 2>&1 &
qwen_pid=$!
```

Inspect the log files while weights load. Check readiness before running inference:

```sh
tail -F "$manual_logs/apertus.log" "$manual_logs/qwen.log"
# Press Ctrl+C to stop watching logs; background servers keep running.
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8001/v1/models
```

With manually started servers, run the existing entry point directly. The automatic
launcher starts its own servers and will reject the ports already occupied by yours:

```sh
CUDA_VISIBLE_DEVICES="${allocated_gpus[3]}" python run_grounded.py \
  --data "$HACK_APERTUS_DATA" --out "$manual_logs/results.json" \
  --apertus-model swiss-ai/Apertus-v1.5-70B \
  --qwen-model Qwen/Qwen3-VL-32B-Instruct \
  --dino-device cuda:0 --sam-device cuda:0

# Stop your manual servers when finished. Leaving the Slurm job also cleans them up.
kill "$apertus_pid" "$qwen_pid"
wait "$apertus_pid" "$qwen_pid"
```

Manual mode leaves readiness, GPU-budget checks, monitoring and server lifecycle
management to you. The automatic launcher provides those when you want a complete
supervised run.

If manual startup crashes, do not start another server until GPU memory is free:

```sh
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

The failed CUDA-capture run left worker processes holding about 85 GiB on each
GPU, causing the next attempt to fail before loading weights. Inspect process
ownership with `ps`, then stop only your failed server and its workers with
SIGTERM; use SIGKILL for confirmed stale processes that remain. Stopping `tail`
does not stop a server. Avoid GPU resets or broad process-kill commands.

The host dataset defaults to `$HOME/hack-apertus/dataset`. Set `HACK_APERTUS_DATA`
or pass `-- --data /mounted/dataset` to change it. Add filesystem mounts for any
dataset/cache/replay path outside `$HOME`/`$SCRATCH`. Baked code runs from
`/opt/hack-apertus`; repository edits require a rebuild. No startup installation
is needed. Check that both servers load and TP2 collectives initialize, that GPU
usage matches the map, that Qwen accepts image/schema requests, and that full
results are produced.

## Configuration, shutdown and replay

Edit `containers/deployment.toml`, use `--config /mounted/deployment.toml`, or set
environment overrides. Every field accepts uppercase `<SECTION>_<FIELD>`:

```sh
export APERTUS_GPUS=0,1 APERTUS_TENSOR_PARALLEL=2 QWEN_GPUS=2 PERCEPTION_GPU=3
export APERTUS_MAX_MODEL_LEN=8192 QWEN_MAX_MODEL_LEN=8192
export APERTUS_MEMORY_UTILIZATION=0.90 QWEN_MEMORY_UTILIZATION=0.90
export RUNTIME_STARTUP_TIMEOUT=600
export APERTUS_EXTRA_ARGS='["--chat-template-content-format","string","--limit-mm-per-prompt","{\"image\":0,\"audio\":0}","--compilation-config.pass_config.fuse_allreduce_rms","false","--enforce-eager"]'
```

Model IDs, ports, revisions, sequence counts, DINO/SAM IDs/revisions and monitoring/
shutdown intervals use the same convention. Extra engine arguments are JSON arrays,
never shell-evaluated. `APERTUS_EXTRA_ARGS` replaces the whole default list, so
preserve the string chat format, media limits and the tested fusion workaround.
The `--enforce-eager` example is an optional diagnostic override; the successful
TP2 run used the fusion workaround without eager mode.
Lower context or disable graph capture to diagnose OOM; raising utilization reduces
headroom. Repeat real requests after changing settings or the image.

The SquashFS image retains the config and sources baked at build time. Rebuild and
export it after source/dependency changes. For a config-only update before a
rebuild, the existing launcher can read the mounted host file:

```sh
bash containers/start_pipeline.sh \
  --config "$HOME/hack-apertus/containers/deployment.toml" -- --limit 1
```

The supervisor starts servers concurrently, waits for `/health` and the expected
`/v1/models` ID, monitors them throughout inference and stops their process groups
(including workers) on success, failure, SIGINT or SIGTERM. It escalates shutdown
to SIGKILL after a timeout. Uncatchable SIGKILL/node loss relies on Slurm cleanup.
Occupied ports, missing CUDA, unsupported implementations, missing shards, readiness
timeouts and child failures produce nonzero exits and useful diagnostics.

Existing pipeline arguments after `--` are forwarded except model/server/device
settings, which belong in deployment config. Replay skips unused servers:

```sh
# Geometry → graphs → Apertus: only Apertus loads.
bash containers/start_pipeline.sh -- --geometry-input /mounted/geometry_results.json
# Geometry → graphs only: no model servers or weights needed.
bash containers/start_pipeline.sh -- --geometry-input /mounted/geometry_results.json --build-only
# Original CPU replay also remains available directly.
python run_grounded.py --geometry-input /mounted/geometry_results.json --build-only \
  --data "$HACK_APERTUS_DATA"
```

Use `--base-dir` when relative mask paths belong to another mounted directory.
The deployment/pipeline never reads challenge ground truth.

## Plan and evaluate from saved grounding artifacts

`run_grounded.py --geometry-input` preserves the grounded pipeline's additional
spatial prompt context. It rebuilds and validates graphs from saved geometry and
SAM mask PNGs, then calls Apertus; extraction, Qwen, DINO and SAM inference are
skipped. `run.py --reuse-scene-graphs` is the earlier Qwen graph workflow and
omits the grounded pipeline's separate spatial context.

With a manually started Apertus server, use a second shell in the same node
allocation. From the **login node**, substituting your running job ID:

```sh
cd "$HOME/hack-apertus"
job_id=YOUR_JOB_ID
srun --jobid="$job_id" --overlap --ntasks=1 \
  --environment=./containers/clariden.edf.toml --pty bash
```

Use the unified EDF for every shell. It supplies `/opt/hack-apertus`, the
application dependencies and `HACK_APERTUS_DATA`/`HACK_APERTUS_OUTPUT`; the
superseded separate EDFs had different paths/caches. In the new container shell:

```sh
cd /opt/hack-apertus || exit 1
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

# Legacy artifact path from the tested replay. For new artifacts use
# $HOME/hack-apertus/outputs/grounded_pipeline/geometry_results.json instead.
geometry_input="$HOME/hack-apertus/geometry_results.json"
run_dir="$HACK_APERTUS_OUTPUT/grounded70b-$(date -u +%Y%m%dT%H%M%S)-$$"

python run_grounded.py \
  --geometry-input "$geometry_input" --data "$HACK_APERTUS_DATA" \
  --apertus-base-url http://127.0.0.1:8000/v1 \
  --apertus-model swiss-ai/Apertus-v1.5-70B \
  --out "$run_dir/results.json"

python starter/evaluate.py "$run_dir/results.json" \
  --data "$HACK_APERTUS_DATA" --report "$run_dir/report.json"
```

Wait for successful server readiness before planning; do not start the automatic
launcher while manual servers occupy its ports. Only the evaluator uses challenge
ground truth. The run writes the submission, raw planning replies, evaluation
report and rebuilt scene graphs into the fresh scratch output directory. Inspect
`results.raw.jsonl` for the exact prompts and validation failures.

## Outputs and remaining validation

All full-pipeline artifacts are written beside the `--out` submission file unless
`--artifact-dir` or `--scene-graph-dir` explicitly redirects them. The standalone
extraction, discovery, DINO, SAM and geometry scripts now default to
`outputs/grounded_pipeline/`, matching `run_grounded.py`. Existing root-level
JSONs are left in place; old artifacts can still be selected with explicit input
flags. SAM records absolute mask/image paths so geometry replay can resolve them
when the metadata JSON lives in the output folder. Rebuild/export the container
to include these updated standalone defaults in its baked source.

**Reusing an output directory overwrites its artifacts.** Submission results and
raw replies are checkpointed during planning, so a failed run can also leave a
partial submission. Graph-only `--build-only` leaves the submission untouched but
rewrites graph/debug files. A subset run replaces its JSON with only the selected
records and can leave older mask/debug images in the same directory.
Dataset files, challenge evaluation code, schemas, and root-level artifacts are
not written by the full pipeline unless an explicit output override targets them.

Use a new directory per run to retain evaluation submissions and their evidence:

```sh
mkdir -p "$HACK_APERTUS_OUTPUT"
run_dir="$HACK_APERTUS_OUTPUT/run-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run_dir"
time bash containers/start_pipeline.sh -- --out "$run_dir/results.json"
python starter/evaluate.py "$run_dir/results.json" --data "$HACK_APERTUS_DATA"
```

For manually started servers, pass the same `--out "$run_dir/results.json"` to
`run_grounded.py` alongside the model/device flags above. A unique `--out` parent
also separates intermediate JSONs, masks, scene graphs and deployment logs; do
not point `--artifact-dir` or `--scene-graph-dir` at a shared old run.
`outputs/` is ignored by Git, so retain or copy any submission/debug evidence
needed for evaluation separately.

The EDF's `HACK_APERTUS_OUTPUT` points to `$SCRATCH/hack-apertus/outputs`, outside
the repository. A root-level `results.json` in your editor is not updated by this
run. Open the scratch run directory directly, or copy it into the repository's
ignored `outputs/` folder for convenient inspection.

With an explicit `--out "$HACK_APERTUS_OUTPUT/full/results.json"`:

```text
$SCRATCH/hack-apertus/
  containers/pipeline-arm64.sqsh
  huggingface/                            # weights + cache manifest
  cache/{vllm,cuda,matplotlib,...}/
  outputs/full/
    results.json
    results.raw.jsonl
    object_extractions.json
    scene_discovery.json
    grounding_detections.json
    grounding_detections_clean.json
    grounding_cleanup_report.json
    segmentation_results.json
    geometry_results.json
    geometry_debug.json
    segmentation/<scene>/<id>.png
    scene_graphs/
    deployment_logs/<timestamp>-<pid>/
      environment.log
      deployment.json
      apertus.log
      qwen.log, 
      pipeline.log
      gpu_memory.csv
      timing.json
```

The GPU CSV records UTC timestamp, index, UUID, name, used/total MiB and utilization
percent for node-visible GPUs; `deployment.json` records the child map, versions,
resolved checkpoints and memory estimates. Timing includes preflight, server startup,
inference and shutdown, with a separate server/pipeline duration. Image build/cache
preparation and Slurm/container startup are excluded from the internal timer; use
outer `time srun ...` if the jury counts those too.

CPU tests cover remapping, configuration failures, protected engine constraints,
cache completeness, memory bounds, real localhost connectivity, readiness timeouts,
child crashes, exit statuses, and forced shutdown, alongside existing perception/
mask/geometry/graph tests. On 2026-10-11, the manual two-GPU Apertus server ran
grounded planning from saved geometry for all 50 instructions and the evaluator
wrote a report. Seven replies failed planning validation and became fail-safe
refusals; model correctness is separate from deployment functionality. The tested
snapshot was `59e744e313e967811aefabde3732609b64201acc`.

Still to validate on Clariden: Qwen3-VL-32B generation on GPU 2, DINO and SAM
inference on GPU 3 in this image, concurrent operation of the complete deployment,
the automatic launcher with real models, and the full build/run budget under
15 minutes. The completed planning replay does not establish these checks.
