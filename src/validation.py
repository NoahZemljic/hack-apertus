import re

from .scene_graph import graph_objects_by_id

ALLOWED_KINDS = {"movable", "container", "surface", "human"}
OBJECT_KEYS = {
    "id", "kind", "label", "color", "bbox_2d", "pick_point_2d",
    "confidence", "relations",
}
RELATION_KEYS = {"inside", "on_top_of", "supports", "held_by_human", "partially_occluded"}


def validate_fields(value, expected: set[str], path: str) -> None:
    """Report the exact missing and unexpected keys at a JSON object path"""
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object with fields {sorted(expected)}, got {value!r}")

    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"{path}: missing fields {missing}; unexpected fields {unexpected}; "
            f"required fields {sorted(expected)}. Include every required field and remove unexpected fields"
        )


def is_integer(value) -> bool:
    """Return whether a value is an integer but not a boolean"""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_pixel(value, name: str, width: int, height: int, length: int) -> list[int]:
    """Validate one pixel point or bounding box against the image bounds"""
    if not isinstance(value, list) or len(value) != length or not all(is_integer(v) for v in value):
        raise ValueError(f"{name}: expected {length} integers, got {value!r}")
    if any(v < 0 for v in value):
        raise ValueError(f"{name} cannot contain negative coordinates")
    if length == 2 and not (value[0] < width and value[1] < height):
        raise ValueError(f"{name}: got {value!r}; require 0 <= x < {width} and 0 <= y < {height}")
    if length == 4 and not (value[0] < value[2] < width and value[1] < value[3] < height):
        raise ValueError(
            f"{name}: got {value!r}; require 0 <= x1 < x2 < {width} "
            f"and 0 <= y1 < y2 < {height} in [x1, y1, x2, y2] order"
        )
    return value


def validate_scene_graph(graph: dict, width: int, height: int) -> dict:
    """Validate scene-graph fields, coordinates, IDs, and relationships"""
    # Reject missing and invented top-level fields before inspecting objects
    validate_fields(graph, {"image", "objects"}, "scene_graph")

    image = graph["image"]
    if image != {"width": width, "height": height}:
        raise ValueError(f"image must be exactly {{'width': {width}, 'height': {height}}}")

    objects = graph["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("objects must be a non-empty list")

    # First pass validates each object independently
    ids = []
    for index, obj in enumerate(objects):
        prefix = f"objects[{index}]"
        validate_fields(obj, OBJECT_KEYS, prefix)
        object_id = obj["id"]
        if not isinstance(object_id, str) or re.fullmatch(r"[a-z][a-z0-9_]*", object_id) is None:
            raise ValueError(f"{prefix}.id must be a lowercase snake_case identifier")
        if not isinstance(obj["kind"], str) or obj["kind"] not in ALLOWED_KINDS:
            raise ValueError(f"{prefix}.kind: expected one of {sorted(ALLOWED_KINDS)}, got {obj['kind']!r}")
        if not isinstance(obj["label"], str) or not obj["label"].strip():
            raise ValueError(f"{prefix}.label must be a non-empty string")
        if obj["color"] is not None and not isinstance(obj["color"], str):
            raise ValueError(f"{prefix}.color must be a string or null")

        bbox = validate_pixel(obj["bbox_2d"], f"{prefix}.bbox_2d", width, height, 4)
        point = validate_pixel(obj["pick_point_2d"], f"{prefix}.pick_point_2d", width, height, 2)
        if not (bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]):
            raise ValueError(f"{prefix}.pick_point_2d must be inside bbox_2d")

        confidence = obj["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"{prefix}.confidence must be between 0 and 1")

        relations = obj["relations"]
        validate_fields(relations, RELATION_KEYS, f"{prefix}.relations")
        if relations["inside"] is not None and not isinstance(relations["inside"], str):
            raise ValueError(f"{prefix}.relations.inside must be a string or null")
        if relations["on_top_of"] is not None and not isinstance(relations["on_top_of"], str):
            raise ValueError(f"{prefix}.relations.on_top_of must be a string or null")
        if (
            not isinstance(relations["supports"], list)
            or not all(isinstance(value, str) for value in relations["supports"])
        ):
            raise ValueError(f"{prefix}.relations.supports must be a list of IDs")
        if len(set(relations["supports"])) != len(relations["supports"]):
            raise ValueError(f"{prefix}.relations.supports contains duplicate IDs")
        for field in ("held_by_human", "partially_occluded"):
            if not isinstance(relations[field], bool):
                raise ValueError(f"{prefix}.relations.{field} must be boolean")
        ids.append(object_id)

    if len(ids) != len(set(ids)):
        raise ValueError("object IDs must be unique")

    # Second pass validates references now that every object ID is known
    known = set(ids)
    for obj in objects:
        refs = [obj["relations"]["inside"], obj["relations"]["on_top_of"], *obj["relations"]["supports"]]
        for ref in (value for value in refs if value is not None):
            if ref not in known:
                raise ValueError(f"{obj['id']}.relations: unknown object {ref!r}; valid IDs are {sorted(known)}")
            if ref == obj["id"]:
                raise ValueError(f"{obj['id']} cannot reference itself")

    # on_top_of and supports must describe both sides of the same contact
    by_id = graph_objects_by_id(graph)
    for obj in objects:
        on_top_of = obj["relations"]["on_top_of"]
        if on_top_of is not None and obj["id"] not in by_id[on_top_of]["relations"]["supports"]:
            raise ValueError(f"{obj['id']} and {on_top_of} have inconsistent support relations")
        for supported in obj["relations"]["supports"]:
            if by_id[supported]["relations"]["on_top_of"] != obj["id"]:
                raise ValueError(f"{obj['id']} and {supported} have inconsistent support relations")

    return graph


def validate_plan(plan: dict, graph: dict) -> dict:
    """Validate a proposed decision and its physical action sequence"""
    validate_fields(plan, {"decision", "steps", "question", "reason"}, "plan")
    decision = plan["decision"]
    if not isinstance(decision, str) or decision not in {"plan", "ask_user", "refuse"}:
        raise ValueError(f"plan.decision: expected plan, ask_user, or refuse; got {decision!r}")
    if not isinstance(plan["steps"], list):
        raise ValueError("steps must be a list")

    known = graph_objects_by_id(graph)

    if decision == "plan":
        if not plan["steps"] or plan["question"] is not None or plan["reason"] is not None:
            raise ValueError("a plan needs steps and null question/reason")

        # Track gripper state so every place follows a valid pick
        holding = False
        for index, step in enumerate(plan["steps"]):
            validate_fields(step, {"action", "object_id"}, f"steps[{index}]")
            if not isinstance(step["action"], str) or step["action"] not in {"pick", "place"}:
                raise ValueError(f"steps[{index}].action: expected pick or place, got {step['action']!r}")
            if not isinstance(step["object_id"], str) or step["object_id"] not in known:
                raise ValueError(
                    f"steps[{index}].object_id: got {step['object_id']!r}; valid IDs are {sorted(known)}"
                )

            obj = known[step["object_id"]]
            if step["action"] == "pick":
                if holding:
                    raise ValueError(f"steps[{index}] picks while the gripper is occupied")
                if obj["kind"] != "movable" or obj["relations"]["held_by_human"]:
                    raise ValueError(f"steps[{index}] tries to pick a non-pickable object")
                holding = True
            else:
                if not holding:
                    raise ValueError(f"steps[{index}] places while the gripper is empty")
                if obj["kind"] == "human":
                    raise ValueError(f"steps[{index}] uses a human as a placement destination")
                holding = False

        if plan["steps"][0]["action"] != "pick":
            raise ValueError("the first step must be pick")

    elif decision == "ask_user":
        if (
            plan["steps"]
            or not isinstance(plan["question"], str)
            or not plan["question"].strip()
            or plan["reason"] is not None
        ):
            raise ValueError("ask_user needs only a non-empty question")
    elif (
        plan["steps"]
        or plan["question"] is not None
        or not isinstance(plan["reason"], str)
        or not plan["reason"].strip()
    ):
        raise ValueError("refuse needs only a non-empty reason")

    return plan
