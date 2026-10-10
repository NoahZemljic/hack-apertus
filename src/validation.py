import json

from jsonschema import Draft202012Validator

from .schemas import load_perception_schema, load_plan_schema, load_scene_graph_schema


def validate_structure(value: dict, schema: dict, name: str) -> dict:
    # JSON syntax: reject values JSON cannot represent, including NaN and infinity.
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: not valid JSON: {error}") from error

    # Schema structure: check required/extra fields, types, array lengths and enums.
    # Scalar limits (e.g. confidence in [0, 1]) are part of the field format.
    errors = list(Draft202012Validator(schema).iter_errors(value))

    # Error reporting: show the exact field path for each independent schema failure.
    if errors:
        details = []
        for error in errors:
            path = "$" + "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}"
                for part in error.absolute_path
            )
            details.append(f"{path}: {error.message}")
        raise ValueError(f"{name} validation failed:\n" + "\n".join(details))
    return value


def validate_perception(detections: dict) -> dict:
    # Detection syntax: labels, four box coordinates, and two target coordinates.
    return validate_structure(detections, load_perception_schema(), "Perception")


def validate_scene_graph(graph: dict) -> dict:
    # Graph syntax only: no reference, cycle, containment, or physical-plausibility checks.
    return validate_structure(graph, load_scene_graph_schema(), "Scene graph")


def validate_plan(plan: dict) -> dict:
    # Plan syntax: the planner's decision and action fields must match its JSON contract.
    return validate_structure(plan, load_plan_schema(), "Plan")
