"""Run Apertus hints → Qwen3-VL-32B → SAM → graphs → validated plans."""

import argparse
from functools import partial
import json
from pathlib import Path
import urllib.error

from starter.evaluate import answer_type
from starter.scene_io import load_scene
from src.apertus import ApertusPlanner
from src.geometry_scene_graph import GeometrySceneGraphBuilder
from src.geometry import GeometryConfig
from src.grounded_perception import GroundedPerceptionConfig, enrich_stacking, generate_geometry
from src.prompting import prepare_planning_prompt
from src.provenance import sha256_text, write_run_manifest
from src.settings import output_root
from src.results import plan_to_result, write_submission_files
from src.visualization import write_scene_overlay


def create_parser() -> argparse.ArgumentParser:
    """Define CLI options for fresh perception, geometry replay and planning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-input", "--input", dest="input", type=Path,
                        help="Start from saved geometry instead of running upstream perception")
    parser.add_argument("--base-dir", type=Path, help="Base for relative mask paths; defaults to input directory")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=output_root() / "grounded_pipeline/results.json")
    parser.add_argument("--artifact-dir", type=Path, help="Intermediate perception files; defaults to output directory")
    parser.add_argument("--scene-graph-dir", type=Path)
    parser.add_argument("--scene", action="append", help="Process only these scenes; repeatable")
    parser.add_argument("--build-only", action="store_true", help="Build graphs and skip final planning; fresh perception still calls Apertus and Qwen")
    parser.add_argument("--no-support-candidates", action="store_true", help="Omit selected geometry support hypotheses")
    parser.add_argument("--apertus-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--apertus-model", "--model", default="swiss-ai/Apertus-v1.5-70B")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--qwen-box-retries", type=int, default=1,
                        help="Format-only retries for Qwen boxes after the initial visual request")
    parser.add_argument("--stacking-source", choices=("qwen", "geometry"), default="qwen",
                        help="Fresh runs: visual Qwen support relations, or legacy geometry hypotheses")
    parser.add_argument("--qwen-stacking-retries", type=int, default=1,
                        help="Format-only retries for the separate Qwen stacking request")
    parser.add_argument("--refresh-stacking", action="store_true",
                        help="With --geometry-input, derive Qwen stacking while reusing saved objects and masks")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--sam-device", help="SAM device, e.g. cuda:0; auto otherwise")
    parser.add_argument("--geometry-config", type=Path, help="JSON GeometryConfig overrides")
    parser.add_argument("--debug-visualizations", action="store_true", help="Also save detector/SAM stage plots (requires matplotlib)")
    parser.add_argument("--no-constrained-decoding", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int, help="Maximum number of instructions, across selected scenes")
    parser.add_argument("--team", default="hack-apertus-grounded")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load scenes, build graphs and save validated Apertus plans.

    ``argv`` is an optional list of CLI arguments; ``None`` reads the process's
    command-line arguments. Saved geometry skips upstream perception, while
    ``--build-only`` skips final planning. Returns 0 on completion; invalid
    arguments, perception/graph failures and connection errors exit via argparse.
    Invalid model plans become logged fail-safe refusals.
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    # Reject invalid settings before loading scenes or calling any models.
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.qwen_box_retries < 0:
        parser.error("--qwen-box-retries must be nonnegative")
    if args.qwen_stacking_retries < 0:
        parser.error("--qwen-stacking-retries must be nonnegative")
    if args.refresh_stacking and (args.input is None or args.stacking_source != "qwen"):
        parser.error("--refresh-stacking requires --geometry-input and --stacking-source qwen")
    if args.base_dir and not args.input:
        parser.error("--base-dir applies only to --geometry-input")
    if not args.data.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data}")
    # Find image-bearing scene folders and apply any explicit scene selection.
    scene_dirs = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").exists())
    if args.scene:
        missing = set(args.scene) - {p.name for p in scene_dirs}
        if missing:
            parser.error(f"Unknown scene names: {sorted(missing)}")
        scene_dirs = [p for p in scene_dirs if p.name in args.scene]
    if not scene_dirs:
        parser.error(f"No scene images found in {args.data}")
    scenes = [(directory, load_scene(directory)) for directory in scene_dirs]
    if args.limit is not None:
        # Keep whole scenes for perception so all their instructions inform discovery.
        # The exact instruction limit is applied later during final planning.
        selected, count = [], 0
        for directory, scene in scenes:
            if count >= args.limit:
                break
            selected.append((directory, scene))
            count += len(scene.instructions)
        scenes = selected
    # Keep intermediate artifacts and graphs beside the submission by default.
    artifact_dir = args.artifact_dir or args.out.parent
    graph_dir = args.scene_graph_dir or args.out.parent / "scene_graphs"
    fresh_perception = args.input is None
    manifest = write_run_manifest(args.out.parent, {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    })
    # A run uses one policy snapshot even if the mounted prompt changes mid-run.
    planning_policy = (args.out.parent / "planning_policy.txt").read_text()
    if fresh_perception or args.refresh_stacking:
        # Fresh run: extraction → Qwen boxes → cleanup → SAM → mask geometry.
        # generate_geometry saves checkpoints and returns the geometry JSON path.
        try:
            overrides = json.loads(args.geometry_config.read_text()) if args.geometry_config else {}
            if not isinstance(overrides, dict):
                raise ValueError("Geometry config must be a JSON object")
            config = GroundedPerceptionConfig(
                apertus_base_url=args.apertus_base_url, apertus_model=args.apertus_model,
                qwen_base_url=args.qwen_base_url,
                timeout=args.timeout, constrained=not args.no_constrained_decoding,
                qwen_box_retries=args.qwen_box_retries,
                stacking_source=args.stacking_source, qwen_stacking_retries=args.qwen_stacking_retries,
                sam_model=args.sam_model, sam_device=args.sam_device,
                debug_visualizations=args.debug_visualizations,
            )
            if fresh_perception:
                args.input = generate_geometry(scenes, artifact_dir, config, GeometryConfig(**overrides))
            else:
                args.input = enrich_stacking(scenes, args.input, artifact_dir, config, base_dir=args.base_dir)
        except (ValueError, OSError, ImportError, KeyError, TypeError, TimeoutError) as error:
            parser.error(f"Perception failed: {error}; diagnostics: {artifact_dir / 'perception.failure.json'}")
    # Both fresh and replay runs load geometry through the same graph builder.
    try:
        builder = GeometrySceneGraphBuilder(args.input, base_dir=args.base_dir,
                                            accept_support_candidates=not args.no_support_candidates)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    missing = [scene.name for _, scene in scenes if scene.name not in builder.records]
    if missing:
        parser.error(f"Geometry results missing scenes: {missing}")

    # Preflight every graph before spending any final planning calls.
    # A build-only run never overwrites an existing submission or pretends to have model results.
    built = []
    graph_dir.mkdir(parents=True, exist_ok=True)
    for _, scene in scenes:
        try:
            graph, spatial, debug = builder.build(scene.name, *scene.size)
        except (ValueError, OSError, KeyError, TypeError) as error:
            (graph_dir / f"{scene.name}.failure.json").write_text(json.dumps({
                "scene": scene.name, "input": str(args.input), "error": str(error),
            }, indent=2) + "\n")
            parser.error(f"{scene.name}: {error}")
        # Save planner input, spatial context and diagnostics separately for inspection.
        for suffix, value in ((".json", graph), (".spatial.json", spatial), (".debug.json", debug)):
            (graph_dir / f"{scene.name}{suffix}").write_text(json.dumps(value, indent=2) + "\n")
        write_scene_overlay(scene.rgb, graph, graph_dir / f"{scene.name}.overlay.jpg")
        built.append((scene, graph, spatial))
        print(f"Validated {scene.name}: {len(graph['objects'])} objects", flush=True)
    if args.build_only:
        # Graph-only runs stop here without writing a submission.
        print(f"Built {len(built)} scene graphs in {graph_dir}; final planning skipped")
        return 0

    results, raw = [], []
    remaining = args.limit
    for scene, graph, spatial in built:
        if remaining is not None and remaining <= 0:
            break
        # Attach this scene's mask-derived spatial context to each planning prompt.
        planner = ApertusPlanner(base_url=args.apertus_base_url, model=args.apertus_model,
                                 timeout=args.timeout, constrained=not args.no_constrained_decoding,
                                 prompt_builder=partial(prepare_planning_prompt, geometry=spatial,
                                                       policy=planning_policy))
        instructions = scene.instructions if remaining is None else scene.instructions[:remaining]
        for instruction_id, _, instruction in instructions:
            prompt = planner.prompt_builder(graph, instruction, scene.gripper_max_opening_mm)
            replies, error = [], None
            try:
                plan, replies = planner.plan(graph, instruction, scene.gripper_max_opening_mm)
                # Use the first picked object's saved safe pixel as the target coordinate.
                result = plan_to_result(instruction_id, plan, graph)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exception:
                # Save available progress and stop if the server cannot be reached.
                if planner.last_trace and planner.last_trace[0].get("messages"):
                    prompt = planner.last_trace[0]["messages"][0]["content"]
                raw.append({"instruction_id": instruction_id, "stage": "planning", "instruction": instruction,
                            "prompt": prompt, "prompt_sha256": sha256_text(prompt),
                            "requests": planner.last_trace,
                            "error": f"{type(exception).__name__}: {exception}"})
                if results:
                    write_submission_files(args.out, args.team, results, raw)
                else:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.with_suffix(".raw.jsonl").write_text(
                        "".join(json.dumps(entry) + "\n" for entry in raw))
                parser.exit(1, f"Cannot obtain an Apertus response at {args.apertus_base_url}: {exception}\n")
            except ValueError as exception:
                # A plan that fails validation produces a refusal and a raw-log error.
                replies = getattr(exception, "replies", [])
                error = f"{type(exception).__name__}: {exception}"
                result = {"instruction_id": instruction_id, "target_pixel": None,
                          "actions": [{"action": "refuse", "reason": "Planning failed validation; no robot motion was issued"}]}
            # Check the final challenge answer format before checkpointing it.
            if planner.last_trace and planner.last_trace[0].get("messages"):
                prompt = planner.last_trace[0]["messages"][0]["content"]
            kind, validation_error = answer_type(result)
            if validation_error:
                raise ValueError(f"{instruction_id}: invalid submission: {validation_error}")
            results.append(result)
            raw.append({"instruction_id": instruction_id, "stage": "planning", "instruction": instruction,
                        "prompt": prompt, "prompt_sha256": sha256_text(prompt),
                        "planning_policy_sha256": manifest["planning_policy_sha256"],
                        "requests": planner.last_trace, "replies": replies, "error": error})
            # Preserve partial progress if a later request fails or the process stops.
            write_submission_files(args.out, args.team, results, raw)
            print(f"[{len(results)}] {instruction_id}: {kind}" + (f" ({error})" if error else ""), flush=True)
        if remaining is not None:
            remaining -= len(instructions)
    print(f"Wrote {len(results)} Apertus results to {args.out}")
    failures = sum(entry.get("error") is not None for entry in raw)
    if failures:
        print(f"Warning: {failures} planning failures produced fail-safe refusals; inspect the raw log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
