import json
from pathlib import Path


SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "schema"


def load_schema(filename: str) -> dict:
    """Read a JSON Schema from the repository's schema directory"""
    # Resolve paths from this module so execution does not depend on the working directory
    return json.loads((SCHEMA_DIRECTORY / filename).read_text())


def load_scene_graph_schema() -> dict:
    """Load the contract supplied to the perception model"""
    return load_schema("scene_graph.schema.json")


def load_plan_schema() -> dict:
    """Load the contract used for Apertus constrained decoding"""
    return load_schema("plan.schema.json")


def load_result_schema() -> dict:
    """Load the contract for the final challenge submission"""
    return load_schema("result.schema.json")
