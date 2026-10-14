"""Check the shared project environment without downloading or loading weights.

Run ``python -m src.check_environment --require-cuda`` inside the GPU container.
Optional local-server checks query model inventories only, never generate text.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import io
import json
from pathlib import Path
import sys
from typing import Any, Callable
import urllib.request


def check_environment(*, profile: str = "vision", require_cuda: bool = False,
                      check_vllm: bool = False,
                      servers: list[tuple[str, str, str]] | None = None) -> dict[str, Any]:
    """Return independent check results; a failed probe cannot hide later failures."""
    results: list[dict[str, Any]] = []

    def probe(name: str, action: Callable[[], Any]) -> None:
        try:
            detail = action()
            results.append({"check": name, "ok": True, "detail": detail})
        except Exception as error:
            results.append({"check": name, "ok": False,
                            "detail": f"{type(error).__name__}: {error}"})

    def python_version() -> str:
        if not (3, 11) <= sys.version_info[:2] < (3, 14):
            raise RuntimeError("Use Python 3.11–3.13 from the compute container")
        return sys.version.split()[0]

    probe("python", python_version)
    if not results[-1]["ok"]:
        return {"ok": False, "profile": profile, "checks": results}

    import tomllib
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    requirements = list(project["project"]["dependencies"])
    if profile == "vision":
        extras = project["project"]["optional-dependencies"]
        # Accelerator packages may be supplied by the container rather than uv.
        requirements += extras["vision"] + extras["accelerator"]

    def dependency(spec: str) -> str:
        from packaging.requirements import Requirement

        requirement = Requirement(spec)
        installed = metadata.version(requirement.name)
        # The official Apertus container ships a compatible development fork.
        if not requirement.specifier.contains(installed, prereleases=True):
            raise RuntimeError(f"{requirement.name} {installed} does not satisfy {requirement.specifier}")
        return f"{requirement.name}=={installed}"

    for requirement in requirements:
        probe(f"dependency: {requirement}", lambda spec=requirement: dependency(spec))

    def core_operations() -> str:
        import cv2
        import numpy as np
        from PIL import Image

        mask = np.zeros((12, 12), np.uint8)
        mask[2:10, 2:10] = 1
        distances = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        if float(distances.max()) != 4:
            raise RuntimeError("OpenCV mask distance transform failed")
        data = io.BytesIO()
        Image.fromarray(mask * 255).save(data, format="PNG")
        data.seek(0)
        if not np.array_equal(np.asarray(Image.open(data)), mask * 255):
            raise RuntimeError("PNG round trip failed")
        return "NumPy/OpenCV distance transform and Pillow PNG round trip"

    probe("mask geometry and PNGs", core_operations)

    if profile == "vision":
        def accelerator_operations() -> dict[str, Any]:
            import numpy as np
            import torch
            import torchvision

            # Importing torchvision alone does not prove its compiled ops work.
            boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])
            scores = torch.tensor([0.9, 0.8])
            keep = torchvision.ops.nms(boxes, scores, 0.5)
            if keep.tolist() != [0] or not np.array_equal(torch.arange(3).numpy(), np.arange(3)):
                raise RuntimeError("Torch/Torchvision/NumPy binary interoperability failed")
            if require_cuda:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable; run inside a GPU allocation with a CUDA-enabled Torch build")
                if torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5).cpu().tolist() != [0]:
                    raise RuntimeError("Torchvision CUDA operator failed")
                product = torch.tensor([[1., 2.]], device="cuda") @ torch.tensor([[1.], [2.]], device="cuda")
                if product.item() != 5:
                    raise RuntimeError("CUDA matrix multiplication failed")
            return {"torch": torch.__version__, "torchvision": torchvision.__version__,
                    "cuda_available": torch.cuda.is_available(), "cuda_verified": require_cuda,
                    "cuda_build": torch.version.cuda}

        probe("Torch/Torchvision operators", accelerator_operations)

        def transformer_operations() -> str:
            import torch
            from PIL import Image
            from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                      GroundingDinoForObjectDetection, Qwen3VLForConditionalGeneration,
                                      Sam2ImageProcessor, Sam2Model, Sam2Processor)

            # Resolving lazy model classes checks their complete import trees.
            assert all(cls is not None for cls in (AutoModelForZeroShotObjectDetection, AutoProcessor,
                       GroundingDinoForObjectDetection, Qwen3VLForConditionalGeneration, Sam2Model))
            processor = Sam2Processor(Sam2ImageProcessor(size={"height": 32, "width": 32}))
            inputs = processor(images=Image.new("RGB", (40, 30)),
                               input_boxes=[[[4, 4, 20, 20]]], return_tensors="pt")
            if tuple(inputs["input_boxes"].shape) != (1, 1, 4):
                raise RuntimeError("Unexpected SAM box prompt shape")
            masks = processor.post_process_masks(torch.ones((1, 1, 3, 8, 8)), inputs["original_sizes"])
            if tuple(masks[0].shape[-2:]) != (30, 40):
                raise RuntimeError("SAM mask restoration failed")
            return "DINO/Qwen/SAM model imports and actual SAM box processing/mask restoration"

        probe("Transformers model integration", transformer_operations)

        def server_client_imports() -> str:
            from openai import OpenAI

            assert OpenAI is not None
            return "OpenAI-compatible client for local discovery/extraction stage scripts"

        probe("local server-client import", server_client_imports)

        def plot_operation() -> str:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            figure = Figure(figsize=(1, 1))
            FigureCanvasAgg(figure)
            figure.subplots().plot([0, 1], [0, 1])
            data = io.BytesIO()
            figure.savefig(data, format="png")
            if not data.getvalue().startswith(b"\x89PNG"):
                raise RuntimeError("Headless PNG rendering failed")
            return "Matplotlib Agg rendering without a display"

        probe("headless debug plots", plot_operation)

    for module in ("run_grounded", "src.masking.sam", "src.scene_discovery"):
        probe(f"project import: {module}", lambda name=module: str(importlib.import_module(name).__file__))

    if check_vllm:
        def vllm_import() -> str:
            from vllm import LLM

            assert LLM is not None
            return metadata.version("vllm")

        probe("vLLM engine import", vllm_import)

    for name, base_url, model in servers or []:
        def server_inventory(url: str = base_url, expected: str = model) -> list[str]:
            request = urllib.request.Request(url.rstrip("/") + "/models",
                                             headers={"Authorization": "Bearer none"})
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read())
            available = [item["id"] for item in body["data"]]
            if expected not in available:
                raise RuntimeError(f"Expected served model {expected!r}; available: {available}")
            return available

        probe(f"local server: {name}", server_inventory)

    return {"ok": all(result["ok"] for result in results), "profile": profile, "checks": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("base", "vision"), default="vision")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-vllm", action="store_true", help="Also import the container-provided inference engine")
    parser.add_argument("--check-servers", action="store_true", help="Query both local model inventories")
    parser.add_argument("--apertus-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--apertus-model", default="swiss-ai/Apertus-v1.5-70B")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)
    if args.require_cuda and args.profile != "vision":
        parser.error("--require-cuda requires --profile vision")
    servers = [("Apertus", args.apertus_base_url, args.apertus_model),
               ("Qwen", args.qwen_base_url, args.qwen_model)] if args.check_servers else None
    report = check_environment(profile=args.profile, require_cuda=args.require_cuda,
                               check_vllm=args.check_vllm, servers=servers)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for result in report["checks"]:
            print(f"{'OK' if result['ok'] else 'FAIL'} {result['check']}: {result['detail']}")
        print("Environment checks passed." if report["ok"] else "Environment checks failed; fix the failures above.")
        if args.profile == "vision" and not args.require_cuda:
            print("CUDA execution was not checked; use --require-cuda on the compute node.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
