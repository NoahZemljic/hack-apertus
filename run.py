import argparse
from pathlib import Path

from starter.evaluate import answer_type
from starter.scene_io import load_scene
from src.apertus import ApertusPlanner
from src.pipeline import RoboticsPipeline
from src.qwen import QwenSceneGraphGenerator
from src.results import write_submission_files


def create_parser() -> argparse.ArgumentParser:
    """Define command-line options for the perception and reasoning pipeline"""
    parser = argparse.ArgumentParser(description="Run Qwen perception followed by Apertus robot planning")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/results.json"))
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--apertus-model", "--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--apertus-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--scene-graph-dir", type=Path)
    parser.add_argument("--reuse-scene-graphs", action="store_true")
    parser.add_argument("--no-constrained-decoding", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--team", default="hack-apertus")
    return parser


def main() -> int:
    """Load scenes, run both models, validate their results, and write the submission"""
    parser = create_parser()
    args = parser.parse_args()

    # Load each image together with its instructions and calibration
    if not args.data.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data}")

    scene_dirs = sorted(
        path
        for path in args.data.iterdir()
        if path.is_dir() and (path / "rgb.png").exists()
    )

    if not scene_dirs:
        parser.error(f"No scene images found in {args.data}")

    scenes = [(scene_dir, load_scene(scene_dir)) for scene_dir in scene_dirs]

    # Qwen converts the image into grounded objects and pixel locations
    vision_model = QwenSceneGraphGenerator(
        model_name=args.qwen_model,
        attention=args.attention,
    )

    # Apertus decides whether to act, ask a question, or refuse
    reasoning_model = ApertusPlanner(
        base_url=args.apertus_base_url,
        model=args.apertus_model,
        timeout=args.timeout,
        constrained=not args.no_constrained_decoding,
    )

    # Run the pipeline
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene_graph_dir = args.scene_graph_dir or args.out.parent / "scene_graphs"

    pipeline = RoboticsPipeline(
        perception=vision_model,
        planner=reasoning_model,
        scene_graph_dir=scene_graph_dir,
        reuse_scene_graphs=args.reuse_scene_graphs,
    )

    results, raw = pipeline.run(scenes, args.limit)

    # Reject malformed submission entries before writing the result file
    invalid = []

    for result in results:
        _, error = answer_type(result)

        if error is not None:
            invalid.append((result.get("instruction_id", "?"), error))

    if invalid:
        parser.error(f"pipeline created invalid final results: {invalid[:3]}")

    # Keep raw responses beside the submission for debugging model failures
    raw_path = write_submission_files(args.out, args.team, results, raw)

    failures = sum(entry.get("error") is not None for entry in raw if entry.get("stage") == "planning")

    print(f"Wrote {len(results)} results to {args.out}")
    print(f"Scene graphs: {scene_graph_dir}")
    print(f"Raw model responses: {raw_path}")

    if failures:
        print(f"Warning: {failures} planner failures produced fail-safe refusals")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
