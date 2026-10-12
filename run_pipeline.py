import argparse
from pathlib import Path

from starter.evaluate import answer_type
from starter.scene_io import load_scene
from src.apertus import APERTUS_MODEL, ApertusPlanner
from src.pipeline import RoboticsPipeline
from src.perception import QwenPerception
from src.vlm import QwenVLM
from src.results import write_submission

def create_parser() -> argparse.ArgumentParser:
    """Define command-line options for the perception and reasoning pipeline"""
    parser = argparse.ArgumentParser(description="Run Qwen perception followed by Apertus robot planning")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/results.json"))
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--apertus-model", "--model", default=APERTUS_MODEL, choices=[APERTUS_MODEL])
    parser.add_argument("--apertus-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--perception-only", action="store_true", help="Create or reuse scene graphs without running Apertus")
    parser.add_argument("--scene", action="append", help="Scene directory name; repeat to select multiple scenes")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=4096)
    parser.add_argument("--qwen-validation-retries", type=int, default=2)
    parser.add_argument(
        "--no-constrained-decoding",
        action="store_true",
        help="Disable Apertus server-side JSON schema constraints",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--team", default="hack-apertus")
    return parser


def main() -> int:
    """Load scenes, run both models, validate their results, and write the submission"""
    parser = create_parser()
    args = parser.parse_args()
    if args.qwen_max_new_tokens < 1 or args.qwen_validation_retries < 0:
        parser.error("Qwen token limit must be positive and retries nonnegative")
    if args.perception_only and args.limit is not None:
        parser.error("--limit counts instructions; use --scene to select perception-only images")

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

    if args.scene:
        missing = set(args.scene) - {path.name for path in scene_dirs}
        if missing:
            parser.error(f"Unknown scenes: {sorted(missing)}")
        scene_dirs = [path for path in scene_dirs if path.name in args.scene]

    scenes = [(scene_dir, load_scene(scene_dir)) for scene_dir in scene_dirs]

    # Qwen converts the image into grounded objects and pixel locations
    vision_model = QwenPerception(QwenVLM(
        model_name=args.qwen_model,
        attention=args.attention,
        max_new_tokens=args.qwen_max_new_tokens,
    ), num_validation_retries=args.qwen_validation_retries)

    # Apertus decides whether to act, ask a question, or refuse
    reasoning_model = ApertusPlanner(
        base_url=args.apertus_base_url,
        model=args.apertus_model,
        timeout=args.timeout,
        constrained=not args.no_constrained_decoding,
    )

    # Create the pipeline
    pipeline = RoboticsPipeline(
        perception=vision_model,
        planner=reasoning_model,
        scene_graph_dir=args.out.parent / "scene_graphs",
    )

    # One scene loop handles both perception-only and complete runs.
    results = pipeline.run(
        scenes,
        limit=args.limit,
        perception_only=args.perception_only,
    )

    if args.perception_only:
        return 0

    # Reject malformed submission entries before writing the result file
    invalid = []

    for result in results:
        _, error = answer_type(result)

        if error is not None:
            invalid.append((result.get("instruction_id", "?"), error))

    if invalid:
        parser.error(f"pipeline created invalid final results: {invalid[:3]}")

    write_submission(args.out, args.team, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
