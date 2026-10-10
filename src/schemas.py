"""Load the small JSON contracts used at each pipeline boundary."""
import json
from pathlib import Path

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "schema"


def load_schema(filename: str) -> dict:
    return json.loads((SCHEMA_DIRECTORY / filename).read_text())


def load_scene_graph_schema() -> dict:
    return load_schema("scene_graph.schema.json")


def load_perception_schema() -> dict:
    return load_schema("perception.schema.json")


def load_plan_schema() -> dict:
    return load_schema("plan.schema.json")


def load_result_schema() -> dict:
    return load_schema("result.schema.json")
