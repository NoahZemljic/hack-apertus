import json
from pathlib import Path

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "schema"


def load_schema(filename: str) -> dict:
    """Read a JSON Schema from the repository's schema directory"""
    # Resolve paths from this module so execution does not depend on the working directory
    return json.loads((SCHEMA_DIRECTORY / filename).read_text())


def load_scene_graph_schema() -> dict:
    """Load the complete scene-graph contract consumed by the planner"""
    return load_schema("scene_graph.schema.json")


def load_perception_schema() -> dict:
    """Derive the model-output contract by excluding code-generated support lists"""
    schema = load_scene_graph_schema()
    schema["title"] = "Perception observations before support derivation"
    schema["$id"] = "see-it-say-it-pick-it/perception.schema.json"
    relations = schema["$defs"]["object"]["properties"]["relations"]
    relations["required"].remove("supports")
    del relations["properties"]["supports"]
    return schema


def load_plan_schema() -> dict:
    """Load the contract used for Apertus constrained decoding"""
    return load_schema("plan.schema.json")


def load_result_schema() -> dict:
    """Load the contract for the final challenge submission"""
    return load_schema("result.schema.json")

def load_object_extraction_schema() -> dict:
    """Load the contract used for Apertus constrained decoding
    in the object extraction step of the pipeline"""
    return load_schema("object_extraction.schema.json")

def load_scene_discovery_schema() -> dict:
    """Load the schema used for Qwen object discovery."""
    return load_schema("scene_discovery.schema.json")
