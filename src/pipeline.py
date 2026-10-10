import json
from pathlib import Path

from .compute_geometry import compute_objects, compute_relations
from .create_scene_graph import create_scene_graph
from .json_tools import parse_json_object
from .perception import PerceptionModel
from .results import plan_to_result
from .validation import validate_perception, validate_scene_graph


class RoboticsPipeline:
    def __init__(
        self,
        perception: PerceptionModel,
        planner,
        scene_graph_dir: Path = Path("outputs/scene_graphs"),
    ) -> None:
        """Connect detection, geometry, scene graphs, and planning."""
        self.perception = perception
        self.planner = planner
        self.scene_graph_dir = scene_graph_dir

    def build_scene_graph(self, scene, scene_dir: Path) -> dict:
        """Reuse the saved graph, or create it if this scene has no graph yet."""
        cache_path = self.scene_graph_dir / f"{scene.name}.json"

        # Always reuse an existing graph.
        if cache_path.exists():
            graph = parse_json_object(cache_path.read_text())
            validate_scene_graph(graph)

            return graph

        # Detect objects, then compute geometry and relations in Python.
        detections, _ = self.perception.detect(scene_dir / "rgb.png", *scene.size)
        validate_perception(detections)

        geometry = compute_objects(scene, detections)
        relations = compute_relations(geometry)
        graph = create_scene_graph(geometry, relations)
        validate_scene_graph(graph)

        # Save only the accepted scene graph.
        self.scene_graph_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(graph, indent=2) + "\n")

        return graph

    def run(
        self,
        scenes: list[tuple[Path, object]],
        limit: int | None = None,
        *,
        perception_only: bool = False,
    ) -> list[dict]:
        """Create each scene graph and optionally plan its instructions."""
        if perception_only and limit is not None:
            raise ValueError("Omit instructions in perception-only mode")

        results = []
        remaining = limit

        for scene_dir, scene in scenes:
            if remaining is not None and remaining <= 0:
                break

            graph = self.build_scene_graph(scene, scene_dir)

            if perception_only:
                continue

            else:
                # Get the instruction
                instructions = scene.instructions[:remaining]

                for instruction_id, _, instruction in instructions:
                    plan, _ = self.planner.create_plan(graph, instruction)
                    result = plan_to_result(instruction_id, plan, graph)
                    results.append(result)

                if remaining is not None:
                    remaining -= len(instructions)

        return results
