"""Load the small JSON contracts used at each pipeline boundary."""
import json
from pathlib import Path

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "schema"


def load_schema(filename: str) -> dict:
    return json.loads((SCHEMA_DIRECTORY / filename).read_text())


def load_scene_graph_schema() -> dict:
    """Load the single SAM-derived scene graph contract."""
    return load_schema("scene_graph.schema.json")


def load_plan_schema() -> dict:
    return load_schema("plan.schema.json")


def load_plan_response_schema(object_ids: list[str] | None = None) -> dict:
    """Let the model assess the reference before committing to a decision.

    Keep the transport schema flat for the decoder. Decision-dependent field
    consistency is still checked against load_plan_schema after generation.
    """
    steps = load_plan_schema()["anyOf"][0]["properties"]["steps"]
    steps["minItems"] = 0
    if object_ids:
        steps["items"]["properties"]["object_id"]["enum"] = sorted(set(object_ids))
        # One transfer per observed object is enough to clear and move a stack.
        # Bound the response so a repeated place cannot consume the token budget.
        steps["maxItems"] = 2 * len(set(object_ids))
    return {
        "type": "object", "additionalProperties": False,
        "required": ["assessment", "decision", "steps", "question", "reason"],
        "properties": {
            "assessment": {"type": "string", "maxLength": 400},
            "decision": {"enum": ["plan", "ask_user", "refuse"]},
            "steps": steps,
            "question": {"type": ["string", "null"]},
            "reason": {"type": ["string", "null"]},
        },
    }



def load_object_extraction_schema() -> dict:
    """Load the contract used for Apertus constrained decoding
    in the object extraction step of the pipeline"""
    return load_schema("object_extraction.schema.json")
