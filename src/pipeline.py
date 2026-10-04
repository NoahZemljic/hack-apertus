import json
from pathlib import Path

from .json_tools import parse_json_object
from .perception import PerceptionGenerationError, PerceptionModel
from .results import plan_to_result
from .validation import validate_scene_graph
from .visualization import write_scene_overlay

class RoboticsPipeline:
    def __init__(
        self,
        perception: PerceptionModel,
        planner,
        scene_graph_dir: Path,
        reuse_scene_graphs: bool = False,
    ) -> None:
        """Connect perception, planning, and scene-graph artifact storage"""
        self.perception = perception
        self.planner = planner
        self.scene_graph_dir = scene_graph_dir
        self.reuse_scene_graphs = reuse_scene_graphs

    def build_scene_graph(self, scene, scene_dir: Path) -> tuple[dict, list[str]]:
        """Load or generate a scene graph and write its inspection artifacts"""
        cache_path = self.scene_graph_dir / f"{scene.name}.json"

        # Reusing a graph avoids repeating perception inference during prompt work
        if self.reuse_scene_graphs and cache_path.exists():
            graph = validate_scene_graph(parse_json_object(cache_path.read_text()), *scene.size)
            replies = ["loaded from cache"]
        else:
            replies = []
            try:
                graph, replies = self.perception.generate_scene_graph(
                    scene_dir / "rgb.png",
                    *scene.size,
                )

                # Enforce the same boundary for every backend, even if it skipped its own validation
                validate_scene_graph(graph, *scene.size)
            except ValueError as error:
                failure = error if isinstance(error, PerceptionGenerationError) else PerceptionGenerationError(
                    f"Perception did not return a valid scene graph: {error}", replies, [str(error)]
                )
                # Save rejected outputs separately so they can never become cached graphs
                self.scene_graph_dir.mkdir(parents=True, exist_ok=True)
                failure_path = self.scene_graph_dir / f"{scene.name}.failure.json"
                failure_path.write_text(json.dumps({
                    "scene": scene.name,
                    "replies": failure.replies,
                    "errors": failure.errors,
                }, indent=2) + "\n")
                if failure is error:
                    raise
                raise failure from error

        self.scene_graph_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(graph, indent=2) + "\n")
        write_scene_overlay(scene.rgb, graph, self.scene_graph_dir / f"{scene.name}.overlay.jpg")
        return graph, replies

    def run(self, scenes: list[tuple[Path, object]], limit: int | None = None) -> tuple[list[dict], list[dict]]:
        """Process every scene instruction and collect results plus raw responses"""
        results = []
        raw = []
        remaining = limit

        for scene_dir, scene in scenes:
            if remaining is not None and remaining <= 0:
                break

            # Runs perception once because all instructions share the same image
            graph, perception_replies = self.build_scene_graph(scene, scene_dir)
            raw.append({"scene": scene.name, "stage": "perception", "replies": perception_replies})
            instructions = scene.instructions if remaining is None else scene.instructions[:remaining]

            for instruction_id, _, instruction in instructions:
                try:
                    plan, replies = self.planner.plan(graph, instruction, scene.gripper_max_opening_mm)
                    result = plan_to_result(instruction_id, plan, graph)
                    error = None

                except Exception as exception:
                    # Invalid plans must never reach the robot as executable actions
                    replies = []
                    error = f"{type(exception).__name__}: {exception}"
                    result = {
                        "instruction_id": instruction_id,
                        "target_pixel": None,
                        "actions": [{"action": "refuse", "reason": "Planning failed validation; no robot motion was issued"}],
                    }

                results.append(result)
                raw.append({"instruction_id": instruction_id, "instruction": instruction,
                            "stage": "planning", "replies": replies, "error": error})

            if remaining is not None:
                remaining -= len(instructions)

        return results, raw
