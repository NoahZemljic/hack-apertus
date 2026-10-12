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

def load_object_extraction_schema() -> dict:
    """Load the contract used for Apertus constrained decoding
    in the object extraction step of the pipeline"""
    return load_schema("object_extraction.schema.json")

def load_scene_discovery_schema() -> dict:
    """Load the schema used for Qwen object discovery."""
    return load_schema("scene_discovery.schema.json")
