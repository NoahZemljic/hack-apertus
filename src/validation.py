import json
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


def validate_scene_graph(graph: dict, width: int, height: int, *, require_supports: bool = True) -> dict:
    """Validate a full graph or raw perception observations before support derivation"""
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
        expected_relations = RELATION_KEYS if require_supports else RELATION_KEYS - {"supports"}
        validate_fields(relations, expected_relations, f"{prefix}.relations")
        if relations["inside"] is not None and not isinstance(relations["inside"], str):
            raise ValueError(f"{prefix}.relations.inside must be a string or null")
        if relations["on_top_of"] is not None and not isinstance(relations["on_top_of"], str):
            raise ValueError(f"{prefix}.relations.on_top_of must be a string or null")
        if (
            not isinstance(relations.get("supports", []), list)
            or not all(isinstance(value, str) for value in relations.get("supports", []))
        ):
            raise ValueError(f"{prefix}.relations.supports must be a list of IDs")
        if len(set(relations.get("supports", []))) != len(relations.get("supports", [])):
            raise ValueError(f"{prefix}.relations.supports contains duplicate IDs")
        for field in ("held_by_human", "partially_occluded"):
            if not isinstance(relations[field], bool):
                raise ValueError(f"{prefix}.relations.{field} must be boolean")
        ids.append(object_id)

    if len(ids) != len(set(ids)):
        raise ValueError("object IDs must be unique")

    # Collect relationship problems together so a retry can repair the whole graph
    by_id = graph_objects_by_id(graph)
    paths = {obj["id"]: f"objects[{index}] ({obj['id']}).relations" for index, obj in enumerate(objects)}
    errors = []
    for obj in objects:
        refs = [(field, obj["relations"][field]) for field in ("inside", "on_top_of")]
        refs.extend((f"supports[{index}]", ref) for index, ref in enumerate(obj["relations"].get("supports", [])))
        for field, ref in refs:
            if ref is None:
                continue
            if ref not in by_id:
                errors.append(
                    f"{paths[obj['id']]}.{field} = {json.dumps(ref)} references an unknown object. "
                    f"Use an existing ID from {json.dumps(ids)} only if supported by the image; "
                    "otherwise use null for inside/on_top_of or remove the supports entry"
                )
            if ref == obj["id"]:
                errors.append(
                    f"{paths[obj['id']]}.{field} references itself. "
                    "An object cannot contain or support itself; correct the reference from the image"
                )

    # Only complete graphs have inverse fields to cross-check
    for obj in objects if require_supports else []:
        object_id = obj["id"]
        on_top_of = obj["relations"]["on_top_of"]
        if on_top_of in by_id and object_id not in by_id[on_top_of]["relations"]["supports"]:
            errors.append(
                f"{paths[object_id]}.on_top_of = {json.dumps(on_top_of)}, but "
                f"{paths[on_top_of]}.supports = {json.dumps(by_id[on_top_of]['relations']['supports'])}. "
                f"If {object_id} rests directly on {on_top_of}, add {json.dumps(object_id)} to "
                f"{paths[on_top_of]}.supports. Otherwise correct {paths[object_id]}.on_top_of "
                "and its inverse to match the image"
            )
        for supported in obj["relations"]["supports"]:
            if supported not in by_id:
                continue
            actual = by_id[supported]["relations"]["on_top_of"]
            if actual != object_id:
                errors.append(
                    f"{paths[object_id]}.supports contains {json.dumps(supported)}, but "
                    f"{paths[supported]}.on_top_of = {json.dumps(actual)}. "
                    f"If {object_id} directly supports {supported}, set {paths[supported]}.on_top_of "
                    f"to {json.dumps(object_id)} and reconcile any previous support claim. "
                    f"Otherwise remove {json.dumps(supported)} from {paths[object_id]}.supports. "
                    f"Do not add {json.dumps(object_id)} to {paths[supported]}.supports to mirror this edge: "
                    "that would reverse the physical support direction"
                )

    # Both containment and physical support must be acyclic, even before derivation
    for field in ("inside", "on_top_of"):
        reported_cycles = set()
        for object_id in ids:
            chain = []
            current = object_id
            while current in by_id:
                if current in chain:
                    cycle = chain[chain.index(current):]
                    key = frozenset(cycle)
                    if key not in reported_cycles:
                        reported_cycles.add(key)
                        assignments = ", ".join(
                            f"{paths[node]}.{field} = {json.dumps(by_id[node]['relations'][field])}"
                            for node in cycle
                        )
                        errors.append(
                            f"Circular {field} chain: {' -> '.join(cycle + [current])}. "
                            f"Conflicting fields: {assignments}. Reinspect the image and correct the "
                            f"false {field} references, using null only when no such detected object exists. "
                            "Do not mirror the relationship in the opposite direction"
                        )
                    break
                chain.append(current)
                current = by_id[current]["relations"][field]

    if errors:
        raise ValueError("Scene relationship validation failed:\n" + "\n".join(
            f"{index}. {error}" for index, error in enumerate(errors, start=1)
        ))

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
