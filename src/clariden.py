"""Supervise the single-container Clariden deployment; inference stays local.

Use containers/start_pipeline.sh --help. Pipeline arguments follow ``--``.
No Torch/CUDA imports occur in this supervisor: GPU contexts belong to children.
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import tomllib
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServerConfig:
    model: str
    gpus: list[int]
    port: int
    tensor_parallel: int
    revision: str = "main"
    max_model_len: int = 8192
    memory_utilization: float = 0.9
    max_num_seqs: int = 1
    extra_args: list[str] | None = None


@dataclass(frozen=True)
class PerceptionConfig:
    gpu: int = 3
    dino_model: str = "IDEA-Research/grounding-dino-base"
    sam_model: str = "facebook/sam2.1-hiera-small"
    dino_revision: str = "main"
    sam_revision: str = "main"


@dataclass(frozen=True)
class RuntimeConfig:
    startup_timeout: float = 600
    shutdown_timeout: float = 15
    monitor_interval: float = 10
    poll_interval: float = 1


@dataclass(frozen=True)
class DeploymentConfig:
    apertus: ServerConfig
    qwen: ServerConfig
    perception: PerceptionConfig
    runtime: RuntimeConfig


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def load_config(path: Path, environment: dict[str, str] | None = None) -> DeploymentConfig:
    env = os.environ if environment is None else environment
    data = tomllib.loads(path.read_text())
    if set(data) - {"apertus", "qwen", "perception", "runtime"}:
        raise ValueError("Unknown deployment config section")
    sections: dict[str, Any] = {}
    for name, cls in (("apertus", ServerConfig), ("qwen", ServerConfig),
                      ("perception", PerceptionConfig), ("runtime", RuntimeConfig)):
        values = dict(data.get(name, {}))
        for field in fields(cls):
            key = f"{name}_{field.name}".upper()
            if key in env:
                raw = env[key]
                if field.name == "gpus":
                    values[field.name] = [int(v.strip()) for v in raw.split(",")]
                elif field.name == "extra_args":
                    values[field.name] = json.loads(raw)
                elif field.name in {"gpu", "port", "tensor_parallel", "max_model_len", "max_num_seqs"}:
                    values[field.name] = int(raw)
                elif field.name in {"memory_utilization", "startup_timeout", "shutdown_timeout", "monitor_interval", "poll_interval"}:
                    values[field.name] = float(raw)
                else:
                    values[field.name] = raw
        sections[name] = cls(**values)
    config = DeploymentConfig(**sections)
    slots: list[int] = []
    for server in (config.apertus, config.qwen):
        if not server.model or not server.revision:
            raise ValueError("Model and revision must not be empty")
        if not isinstance(server.gpus, list) or not server.gpus or any(type(v) is not int or v < 0 for v in server.gpus):
            raise ValueError("Server GPUs must be nonnegative allocation positions")
        slots.extend(server.gpus)
        if server.tensor_parallel != len(server.gpus):
            raise ValueError("tensor_parallel must equal the number of assigned GPUs")
        if not 0 < server.memory_utilization < 1 or not math.isfinite(server.memory_utilization):
            raise ValueError("memory_utilization must be between 0 and 1")
        if not 1024 <= server.port <= 65535 or server.max_model_len < 1 or server.max_num_seqs < 1:
            raise ValueError("Invalid server port, context length, or sequence count")
        if server.extra_args is not None and (not isinstance(server.extra_args, list) or
                                              any(not isinstance(v, str) for v in server.extra_args)):
            raise ValueError("extra_args must be a JSON/TOML list of strings")
        # Extra engine flags may tune compilation, but cannot override isolation.
        protected = {"--host", "--port", "--model", "--served-model-name", "--tensor-parallel-size",
                     "--max-model-len", "--gpu-memory-utilization", "--max-num-seqs", "--api-key"}
        if any(v.split("=", 1)[0] in protected for v in server.extra_args or []):
            raise ValueError("Configure host/port/model/GPU/context/memory through deployment fields")
    if type(config.perception.gpu) is not int or config.perception.gpu < 0:
        raise ValueError("Perception GPU must be a nonnegative allocation position")
    slots.append(config.perception.gpu)
    if len(set(slots)) != len(slots):
        raise ValueError("GPU allocations must be disjoint and contain no duplicate positions")
    if config.apertus.port == config.qwen.port:
        raise ValueError("Model servers must use separate ports")
    if any(not math.isfinite(v) or v <= 0 for v in asdict(config.runtime).values()):
        raise ValueError("Runtime intervals/timeouts must be finite and positive")
    return config


def gpu_assignments(config: DeploymentConfig, visible: str | None, count: int = 4) -> dict[str, str]:
    """Map allocation positions to Slurm's original numeric IDs or GPU UUIDs."""
    devices = [v.strip() for v in visible.split(",")] if visible is not None else [str(i) for i in range(count)]
    if not devices or any(not v or v == "-1" for v in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES is empty, disabled, or duplicated")
    result = {}
    for name, positions in (("apertus", config.apertus.gpus), ("qwen", config.qwen.gpus),
                            ("pipeline", [config.perception.gpu])):
        if max(positions) >= len(devices):
            raise ValueError(f"{name} requires allocation positions {positions}; visible allocation: {devices}")
        result[name] = ",".join(devices[i] for i in positions)
    return result


def child_environment(base: dict[str, str], devices: str, cache_name: str) -> dict[str, str]:
    result = dict(base)
    result.update(CUDA_VISIBLE_DEVICES=devices, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                  VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTHONUNBUFFERED="1")
    if "VLLM_CACHE_ROOT" in base:
        result["VLLM_CACHE_ROOT"] = str(Path(base["VLLM_CACHE_ROOT"]) / cache_name)
    # vLLM children must not inherit a Slurm-distributed Torch rank/world.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        result.pop(key, None)
    # These private localhost servers use the pipeline's existing unauthenticated
    # clients, independently of credentials for other services in the host shell.
    result.pop("VLLM_API_KEY", None)
    return result


def engine_report() -> dict[str, Any]:
    """Resolve actual lazy model classes without downloading model checkpoints."""
    from transformers import (Apertus1p5ForConditionalGeneration, Qwen3VLForConditionalGeneration,
                              GroundingDinoForObjectDetection, Sam2Model)
    from vllm.model_executor.models import ModelRegistry

    names = ["Apertus1p5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"]
    for name, module in zip(names, ("apertus_mm", "qwen3_vl")):
        if name not in ModelRegistry.get_supported_archs():
            raise RuntimeError(f"This engine does not register {name}; use the pinned Swiss AI release")
        # Import implementations directly: public registry inspection requires a
        # full ModelConfig in this release, which can fetch model metadata.
        if getattr(importlib.import_module(f"vllm.model_executor.models.{module}"), name) is None:
            raise RuntimeError(f"vLLM could not import {name}")
    assert all(cls is not None for cls in (Apertus1p5ForConditionalGeneration,
               Qwen3VLForConditionalGeneration, GroundingDinoForObjectDetection, Sam2Model))
    return {"machine": platform.machine(), "architectures": names,
            "versions": {name: metadata.version(name) for name in ("vllm", "transformers", "torch", "torchvision")},
            "base_image": os.environ.get("HACK_APERTUS_BASE_IMAGE", "not recorded")}


def resolve_snapshot(model: str, revision: str, *, download: bool = False) -> Path:
    if Path(model).is_dir():
        return Path(model).resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model, revision=revision, local_files_only=not download)).resolve()
    except Exception as error:
        raise RuntimeError(f"Cannot resolve {model}@{revision} in HF_HOME. Prepare its cache first "
                           f"with --prepare-cache (Apertus requires accepted Hub access). {error}") from error


def snapshot_metadata(path: Path) -> dict[str, Any]:
    """Check complete local shards; estimate stored weights without loading tensors."""
    config = json.loads((path / "config.json").read_text())
    index = path / "model.safetensors.index.json"
    if index.exists():
        contents = json.loads(index.read_text())
        filenames = sorted(set(contents["weight_map"].values()))
        weight_bytes = contents.get("metadata", {}).get("total_size")
    else:
        filenames = [p.name for p in path.glob("*.safetensors")]
        weight_bytes = None
    if not filenames:
        raise RuntimeError(f"No safetensors weights in {path}")
    for name in filenames:
        weight = path / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not weight.is_file() or weight.stat().st_size == 0:
            raise RuntimeError(f"Missing, empty, or invalid cached shard: {weight}")
    if weight_bytes is None:
        weight_bytes = sum((path / name).stat().st_size for name in filenames)
    return {"path": str(path), "architectures": config.get("architectures", []),
            "weight_bytes": weight_bytes, "shards": len(filenames), "config": config}


def memory_estimate(model: dict[str, Any], server: ServerConfig, total_bytes: list[int]) -> dict[str, Any]:
    """Optimistic TP weight/KV lower bounds, NOT a proof that the model fits."""
    text = model["config"].get("text_config", model["config"])
    kv_bytes = None
    if all(key in text for key in ("num_hidden_layers", "num_key_value_heads")):
        head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
        # BF16 K and V; KV heads can be replicated when TP exceeds their count.
        heads_per_rank = math.ceil(text["num_key_value_heads"] / server.tensor_parallel)
        kv_bytes = 2 * 2 * text["num_hidden_layers"] * heads_per_rank * head_dim * server.max_model_len * server.max_num_seqs
    weights = model["weight_bytes"] / server.tensor_parallel
    budgets = [total * server.memory_utilization for total in total_bytes]
    estimate = {"weights_per_rank_GiB": weights / 2**30,
                "kv_per_rank_GiB": kv_bytes / 2**30 if kv_bytes is not None else None,
                "budgets_GiB": [v / 2**30 for v in budgets],
                "note": "Optimistic only: replicated multimodal weights, workspaces, activations, CUDA graphs and NCCL add memory"}
    if any(weights + (kv_bytes or 0) >= budget for budget in budgets):
        raise RuntimeError(f"Weight/KV lower bound already exceeds the configured GPU budget: {estimate}")
    return estimate


def server_command(server: ServerConfig, model_path: str) -> list[str]:
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
            "--served-model-name", server.model, "--host", "127.0.0.1", "--port", str(server.port),
            "--tensor-parallel-size", str(server.tensor_parallel), "--dtype", "bfloat16",
            "--max-model-len", str(server.max_model_len), "--gpu-memory-utilization", str(server.memory_utilization),
            "--max-num-seqs", str(server.max_num_seqs), *(server.extra_args or [])]


def pipeline_command(config: DeploymentConfig, extra: list[str], paths: dict[str, str] | None = None) -> list[str]:
    paths = paths or {}
    output = Path(os.environ.get("HACK_APERTUS_OUTPUT", "outputs/clariden"))
    command = [sys.executable, str(ROOT / "run_grounded.py"),
               "--data", os.environ.get("HACK_APERTUS_DATA", "dataset"),
               "--out", str(output / "results.json"),
               "--apertus-base-url", f"http://127.0.0.1:{config.apertus.port}/v1",
               "--apertus-model", config.apertus.model,
               "--qwen-base-url", f"http://127.0.0.1:{config.qwen.port}/v1",
               "--qwen-model", config.qwen.model,
               "--dino-model", paths.get("dino", config.perception.dino_model),
               "--sam-model", paths.get("sam", config.perception.sam_model),
               "--dino-device", "cuda:0", "--sam-device", "cuda:0"]
    protected = {"--apertus-base-url", "--apertus-model", "--model", "--qwen-base-url", "--qwen-model",
                 "--dino-model", "--sam-model", "--dino-device", "--sam-device"}
    if any(arg.split("=", 1)[0] in protected for arg in extra):
        raise ValueError("Use deployment config/environment variables for models and devices")
    return command + extra


def parse_pipeline(command: list[str]) -> Any:
    from run_grounded import create_parser

    # Disable abbreviations so isolation flags cannot be overridden indirectly.
    parser = create_parser()
    parser.allow_abbrev = False
    return parser.parse_args(command[2:])


def validate_pipeline(command: list[str]) -> Any:
    args = parse_pipeline(command)
    if args.limit is not None and args.limit < 1 or args.timeout <= 0:
        raise ValueError("Pipeline limit and timeout must be positive")
    if not 0 <= args.box_threshold <= 1 or not 0 <= args.text_threshold <= 1:
        raise ValueError("Pipeline detection thresholds must lie in [0,1]")
    if args.qwen_box_retries < 0:
        raise ValueError("Qwen box retries must be nonnegative")
    if not args.input and args.box_detector == "qwen" and args.query_source != "qwen_objects":
        raise ValueError("--query-source applies only to --box-detector dino")
    if args.base_dir and not args.input:
        raise ValueError("--base-dir requires --geometry-input")
    if not args.data.is_dir():
        raise ValueError(f"Dataset directory does not exist: {args.data}; mount it and set HACK_APERTUS_DATA or --data")
    scenes = {p.name for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").is_file()}
    if not scenes or args.scene and set(args.scene) - scenes:
        raise ValueError("No matching scene images in dataset")
    if args.input and not args.input.is_file():
        raise ValueError(f"Geometry replay input not found: {args.input}")
    if args.geometry_config:
        from src.geometry import GeometryConfig
        GeometryConfig(**json.loads(args.geometry_config.read_text()))
    return args


def needed_servers(args: Any, *, smoke: bool = False) -> list[str]:
    if smoke or args.input is None:
        return ["apertus", "qwen"]
    return [] if args.build_only else ["apertus"]


def perception_model_definitions(config: DeploymentConfig, box_detector: str) -> list[tuple[str, str, str]]:
    """Qwen boxes use the existing server; only DINO needs local detector weights."""
    if box_detector not in {"dino", "qwen"}:
        raise ValueError(f"Unknown box detector: {box_detector}")
    definitions = []
    if box_detector == "dino":
        definitions.append(("dino", config.perception.dino_model, config.perception.dino_revision))
    definitions.append(("sam", config.perception.sam_model, config.perception.sam_revision))
    return definitions


def model_definitions(config: DeploymentConfig, args: Any, *, smoke: bool = False) -> list[tuple[str, str, str]]:
    definitions = [(name, getattr(config, name).model, getattr(config, name).revision)
                   for name in needed_servers(args, smoke=smoke)]
    if args.input is None:
        definitions += perception_model_definitions(config, args.box_detector)
    return definitions


def port_available(port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Localhost port {port} is already occupied; choose another port or stop its owner") from error


def server_ready(port: int, model: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if response.status != 200:
                return False
        with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            available = [item["id"] for item in json.load(response)["data"]]
        return model in available
    except (OSError, ValueError, KeyError, TimeoutError):
        return False


@dataclass
class Child:
    name: str
    process: subprocess.Popen
    log_path: Path
    stream: Any


class Supervisor:
    """Own process groups including vLLM workers, and always reap children."""

    def __init__(self, log_dir: Path, runtime: RuntimeConfig):
        self.log_dir = log_dir
        self.runtime = runtime
        self.children: list[Child] = []
        self.last_monitor = -math.inf

    def start(self, name: str, command: list[str], env: dict[str, str]) -> Child:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{name}.log"
        stream = path.open("w")
        log(f"Starting {name}: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} {shlex.join(command)}; log={path}")
        try:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            stream.close()
            raise
        child = Child(name, process, path, stream)
        self.children.append(child)
        return child

    def check_alive(self, children: list[Child]) -> None:
        for child in children:
            status = child.process.poll()
            if status is not None:
                tail = "\n".join(child.log_path.read_text(errors="replace").splitlines()[-35:])
                raise RuntimeError(f"{child.name} exited unexpectedly ({status}). {child.log_path}\n{tail}")

    def monitor(self) -> None:
        if time.monotonic() - self.last_monitor < self.runtime.monitor_interval:
            return
        self.last_monitor = time.monotonic()
        command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                   "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if result.returncode:
                log(f"GPU monitor unavailable: {result.stderr.strip()}")
                return
            with (self.log_dir / "gpu_memory.csv").open("a") as stream:
                for line in result.stdout.splitlines():
                    stream.write(f"{datetime.now(timezone.utc).isoformat()},{line}\n")
            log("GPU memory MiB (index, UUID, name, used, total, utilization): " + " | ".join(result.stdout.splitlines()))
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"GPU monitoring warning: {error}")

    def wait_ready(self, servers: list[tuple[Child, ServerConfig]]) -> None:
        deadline = time.monotonic() + self.runtime.startup_timeout
        pending = list(servers)
        last_progress = -math.inf
        while pending:
            self.check_alive([child for child, _ in servers])
            for child, config in list(pending):
                if server_ready(config.port, config.model):
                    log(f"{child.name} ready at http://127.0.0.1:{config.port}/v1; served model={config.model}")
                    pending.remove((child, config))
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Servers not ready after {self.runtime.startup_timeout}s: "
                                   f"{[child.name for child, _ in pending]}; inspect logs in {self.log_dir}")
            self.monitor()
            if time.monotonic() - last_progress >= 30:
                log(f"Waiting for {[child.name for child, _ in pending]}; deadline in {max(0, deadline-time.monotonic()):.0f}s")
                last_progress = time.monotonic()
            time.sleep(self.runtime.poll_interval)

    def wait_pipeline(self, pipeline: Child, servers: list[Child]) -> int:
        while pipeline.process.poll() is None:
            self.check_alive(servers)
            self.monitor()
            time.sleep(self.runtime.poll_interval)
        # A server crashing at the same time must not be reported as success.
        self.check_alive(servers)
        status = pipeline.process.returncode
        log(f"Pipeline exited with status {status}; output log={pipeline.log_path}")
        if status:
            log("\n".join(pipeline.log_path.read_text(errors="replace").splitlines()[-35:]))
        return status if status >= 0 else 128 - status

    def close(self) -> None:
        # Signal groups even if their leader has exited: engine workers can survive it.
        for child in reversed(self.children):
            try:
                os.killpg(child.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + self.runtime.shutdown_timeout
        for child in reversed(self.children):
            try:
                child.process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for child in reversed(self.children):
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.process.wait()
            child.stream.close()
        log("All deployment process groups stopped")


def gpu_probe() -> list[dict[str, Any]]:
    """Executed in a subprocess so the supervisor never holds GPU allocations."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: launch inside the four-GPU EDF allocation")
    return [{"position": i, "name": torch.cuda.get_device_properties(i).name,
             "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
             "capability": list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]


def subprocess_json(arguments: list[str], env: dict[str, str]) -> Any:
    result = subprocess.run([sys.executable, "-m", "src.clariden", *arguments], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Preflight failed: {' '.join(arguments)}\n{result.stdout}\n{result.stderr}")
    # vLLM may log before the final, prefixed JSON record.
    records = [line.removeprefix("REPORT:") for line in result.stdout.splitlines() if line.startswith("REPORT:")]
    if not records:
        raise RuntimeError(f"No preflight report:\n{result.stdout}\n{result.stderr}")
    return json.loads(records[-1])


def run(config: DeploymentConfig, extra: list[str], *, dry_run: bool = False,
        smoke: bool = False, preflight_only: bool = False, log_dir: Path | None = None) -> int:
    deployment_started = time.monotonic()
    command = pipeline_command(config, extra)
    assignments = gpu_assignments(config, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if dry_run:
        print(json.dumps({"config": asdict(config), "gpu_assignments": assignments,
                          "servers": {name: server_command(server, server.model) for name, server in
                                      (("apertus", config.apertus), ("qwen", config.qwen))},
                          "pipeline": command, "offline": True}, indent=2))
        return 0
    args = validate_pipeline(command)
    names = needed_servers(args, smoke=smoke)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    logs = log_dir or args.out.parent / "deployment_logs" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ)
    if "HF_HOME" not in base:
        raise ValueError("Set HF_HOME to a mounted persistent model cache (the Clariden EDF does this)")
    for key in ("HF_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH"):
        if key in base:
            Path(base[key]).mkdir(parents=True, exist_ok=True)
    # HTTP clients in the existing pipeline must always connect directly locally.
    base["NO_PROXY"] = base["no_proxy"] = "localhost,127.0.0.1,::1"
    base.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1")
    log(f"Single-container deployment; GPU assignments={assignments}; persistent cache={base['HF_HOME']}; logs={logs}")
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("Clariden deployment requires an ARM64 node")
    gpus = subprocess_json(["--gpu-probe"], base)
    required_positions = config.apertus.gpus + config.qwen.gpus + [config.perception.gpu]
    if max(required_positions) >= len(gpus):
        raise RuntimeError(f"Required GPU position not visible: {gpus}")
    if any(tuple(gpus[i]["capability"]) < (9, 0) for i in required_positions):
        raise RuntimeError(f"Expected GH200/Hopper or newer GPUs: {gpus}")
    log(f"Visible GPU inventory: {gpus}")
    engine = subprocess_json(["--check-engine"], child_environment(base, assignments["pipeline"], "preflight"))
    log(f"Engine compatibility: {engine}")
    checker = subprocess.run([sys.executable, "-m", "src.check_environment", "--require-cuda", "--check-vllm"],
                             cwd=ROOT, env=child_environment(base, assignments["pipeline"], "preflight"),
                             capture_output=True, text=True, timeout=180)
    (logs / "environment.log").write_text(checker.stdout + checker.stderr)
    if checker.returncode:
        raise RuntimeError(f"Environment/CUDA preflight failed:\n{checker.stdout}\n{checker.stderr}")
    models: dict[str, Any] = {}
    estimates = {}
    for name, model, revision in model_definitions(config, args, smoke=smoke):
        models[name] = snapshot_metadata(resolve_snapshot(model, revision))
        if name in names:
            server = getattr(config, name)
            registered = engine["architectures"]
            if not set(models[name]["architectures"]) & set(registered):
                raise RuntimeError(f"Unsupported cached architecture for {name}: {models[name]['architectures']}")
            estimates[name] = memory_estimate(models[name], server, [gpus[i]["memory_bytes"] for i in server.gpus])
            log(f"{name} optimistic memory estimate: {estimates[name]}")
            port_available(server.port)
    paths = {name: model["path"] for name, model in models.items()}
    command = pipeline_command(config, extra, paths)
    (logs / "deployment.json").write_text(json.dumps({"config": asdict(config), "gpu_assignments": assignments,
           "gpus": gpus, "engine": engine, "models": models, "memory_estimates": estimates,
           "pipeline": command, "servers_needed": names}, indent=2) + "\n")
    if preflight_only:
        log("Preflight passed; no model loaded and no servers launched")
        return 0
    supervisor = Supervisor(logs, config.runtime)
    started = time.monotonic()
    try:
        servers = [(supervisor.start(name, server_command(getattr(config, name), paths[name]),
                     child_environment(base, assignments[name], name)), getattr(config, name)) for name in names]
        supervisor.wait_ready(servers)
        if smoke:
            log("Smoke test passed: GPU/import/configuration/cache checks and both local server inventories; no full pipeline inference")
            return 0
        pipeline = supervisor.start("pipeline", command, child_environment(base, assignments["pipeline"], "pipeline"))
        return supervisor.wait_pipeline(pipeline, [child for child, _ in servers])
    finally:
        supervisor.close()
        elapsed = time.monotonic() - started
        total_elapsed = time.monotonic() - deployment_started
        (logs / "timing.json").write_text(json.dumps({"server_start_pipeline_shutdown_seconds": elapsed,
                      "total_deployment_seconds": total_elapsed, "under_15_minutes": total_elapsed < 900}, indent=2) + "\n")
        log(f"Server startup + pipeline + shutdown elapsed: {elapsed:.1f}s (build/preflight excluded)")
        log(f"Total deployment elapsed including preflight: {total_elapsed:.1f}s (image build excluded)")


def prepare_cache(config: DeploymentConfig, box_detector: str = "dino") -> None:
    """Explicit preparation operation, separate from offline startup/build."""
    if not os.environ.get("HF_HOME"):
        raise ValueError("Set HF_HOME to a persistent mounted directory before preparing models")
    # These flags must be set before importing huggingface_hub.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    records = {}
    definitions = [("apertus", config.apertus.model, config.apertus.revision),
                   ("qwen", config.qwen.model, config.qwen.revision)]
    definitions += perception_model_definitions(config, box_detector)
    for name, model, revision in definitions:
        log(f"Preparing {name}: {model}@{revision} into HF_HOME={os.environ['HF_HOME']}")
        records[name] = snapshot_metadata(resolve_snapshot(model, revision, download=True))
    path = Path(os.environ["HF_HOME"]) / "hack_apertus_cache_manifest.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    log(f"Complete cache manifest: {path}; pin snapshot commit revisions for repeatable runs")


class Interrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "containers/deployment.toml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands and isolation; no GPU/import/cache checks")
    mode.add_argument("--preflight-only", action="store_true", help="Check allocation/imports/cache/memory/config; do not load models")
    mode.add_argument("--smoke-test", action="store_true", help="Preflight and launch both servers; verify inventories, then stop")
    mode.add_argument("--prepare-cache", action="store_true", help="Explicitly download weights once into persistent HF_HOME")
    mode.add_argument("--check-engine", action="store_true", help="Check both model implementations without weights")
    mode.add_argument("--gpu-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="Arguments for run_grounded.py after --")
    args = parser.parse_args(argv)
    extra = args.pipeline_args[1:] if args.pipeline_args[:1] == ["--"] else args.pipeline_args
    old_handlers = {}

    def stop(signum: int, frame: Any) -> None:
        # Further signals must not interrupt process-group cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise Interrupted(signum)

    try:
        if args.check_engine:
            print("REPORT:" + json.dumps(engine_report()))
            return 0
        if args.gpu_probe:
            print("REPORT:" + json.dumps(gpu_probe()))
            return 0
        config = load_config(args.config)
        if args.prepare_cache:
            pipeline_args = parse_pipeline(pipeline_command(config, extra))
            prepare_cache(config, pipeline_args.box_detector)
            return 0
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, stop)
        return run(config, extra, dry_run=args.dry_run, smoke=args.smoke_test,
                   preflight_only=args.preflight_only, log_dir=args.log_dir)
    except Interrupted as error:
        log(f"Stopped by signal {error.signum}")
        return 128 + error.signum
    except (Exception,) as error:
        log(f"DEPLOYMENT ERROR: {type(error).__name__}: {error}")
        return 1
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
