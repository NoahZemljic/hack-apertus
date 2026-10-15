import json

from jsonschema import Draft202012Validator

from .scene_graph import supporting_ids
from .schemas import load_plan_schema, load_scene_graph_schema


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


def validate_scene_graph(graph: dict, width: int | None = None,
                         height: int | None = None) -> dict:
    """Validate the SAM-derived graph and its image coordinates and references."""
    validate_structure(graph, load_scene_graph_schema(), "Grounded scene graph")
    image = graph["image"]
    width = image["width"] if width is None else width
    height = image["height"] if height is None else height
    if image != {"width": width, "height": height}:
        raise ValueError("Grounded scene graph image dimensions do not match RGB image")
    objects = {obj["id"]: obj for obj in graph["objects"]}
    if len(objects) != len(graph["objects"]):
        raise ValueError("Grounded scene graph contains duplicate IDs")
    expected_supports = {oid: [] for oid in objects}
    for oid, obj in objects.items():
        x1, y1, x2, y2 = obj["bbox_2d"]
        u, v = obj["pick_point_2d"]
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            raise ValueError(f"{oid}: bbox_2d must use inclusive in-image bounds")
        if not (x1 <= u <= x2 and y1 <= v <= y2):
            raise ValueError(f"{oid}: pick_point_2d lies outside bbox_2d")
        for field in ("inside", "on_top_of"):
            value = obj["relations"][field]
            targets = supporting_ids(value) if field == "on_top_of" else ([] if value is None else [value])
            for target in targets:
                if target not in objects or target == oid:
                    raise ValueError(f"{oid}: {field} references an unknown object or itself")
                if objects[target]["kind"] == "human":
                    raise ValueError(f"{oid}: {field} targets a human")
                if field == "inside" and objects[target]["kind"] != "container":
                    raise ValueError(f"{oid}: inside must target a container")
                if field == "on_top_of":
                    expected_supports[target].append(oid)
        for target in obj["relations"].get("occluded_by", []):
            if target not in objects or target == oid:
                raise ValueError(f"{oid}: occluded_by references an unknown object or itself")
    for oid, obj in objects.items():
        if sorted(obj["relations"]["supports"]) != sorted(expected_supports[oid]):
            raise ValueError(f"{oid}: supports must be the inverse of on_top_of")
    # Check each hierarchy and their union; a container cannot be under its contents.
    for fields in (("inside",), ("on_top_of",), ("inside", "on_top_of")):
        completed = set()
        active = set()

        def visit(oid):
            if oid in active:
                raise ValueError(f"{oid}: relationship cycle in {fields}")
            if oid in completed:
                return
            active.add(oid)
            for field in fields:
                value = objects[oid]["relations"][field]
                targets = supporting_ids(value) if field == "on_top_of" else ([] if value is None else [value])
                for target in targets:
                    visit(target)
            active.remove(oid)
            completed.add(oid)

        for oid in objects:
            visit(oid)
    return graph


def validate_plan(plan: dict, graph: dict | None = None) -> dict:
    """Validate the decision contract and, when given a graph, mechanical actions.

    This does not interpret the instruction or invent missing support observations.
    It only checks exact IDs, gripper state and the supplied direct-support edges.
    """
    validate_structure(plan, load_plan_schema(), "Plan")
    if graph is None or plan["decision"] != "plan":
        return plan
    objects = {obj["id"]: obj for obj in graph["objects"]}
    blockers = {oid: set(obj.get("relations", {}).get("supports", [])) for oid, obj in objects.items()}
    held = None
    for index, step in enumerate(plan["steps"]):
        oid = step["object_id"]
        if oid not in objects:
            raise ValueError(f"Step {index + 1}: unknown object ID {oid!r}; use an exact supplied ID")
        obj = objects[oid]
        if step["action"] == "pick":
            if held is not None:
                raise ValueError(f"Step {index + 1}: cannot pick {oid} while holding {held}; place it first")
            if obj.get("kind") != "movable" or obj.get("relations", {}).get("held_by_human"):
                raise ValueError(f"Step {index + 1}: {oid} is not an available movable source")
            if blockers[oid]:
                raise ValueError(f"Step {index + 1}: clear objects resting on {oid} first: {sorted(blockers[oid])}")
            held = oid
            for resting in blockers.values():
                resting.discard(oid)
        else:
            if held is None:
                raise ValueError(f"Step {index + 1}: cannot place with an empty gripper")
            if oid == held:
                raise ValueError(f"Step {index + 1}: place names held source {held}, not a destination")
            if obj.get("kind") == "human":
                raise ValueError(f"Step {index + 1}: cannot place on a human")
            if obj.get("kind") == "movable":
                blockers[oid].add(held)
            held = None
    return plan
