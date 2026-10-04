import json
import math
from collections import Counter

from jsonschema import Draft202012Validator

from .scene_graph import graph_objects_by_id
from .schemas import load_perception_schema, load_scene_graph_schema

# These categories describe manipulation roles, not whether an object is hollow
CATEGORY_KINDS = {
    "tray": "container", "plate": "surface", "saucer": "surface",
    "hand": "human", "mug": "movable", "cup": "movable", "bowl": "movable",
    "gear": "movable", "cube": "movable", "banana": "movable", "mandarin": "movable",
    "bracket": "movable", "part": "movable", "component": "movable",
}
RECEPTACLE_LABELS = {"mug", "cup", "bowl"}
BACKGROUND_LABELS = {"table", "tabletop", "floor", "background", "workbench"}
CONTAINMENT_MARGIN_FRACTION = 0.02

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
    """Check for an actual integer rather than a boolean or integral float"""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_pixel(value, name: str, width: int, height: int, length: int) -> list[int]:
    """Validate pixel indices or half-open bounding-box edges"""
    if not isinstance(value, list) or len(value) != length or not all(is_integer(v) for v in value):
        raise ValueError(f"{name}: expected {length} integers, got {value!r}")
    if length == 2 and not (0 <= value[0] < width and 0 <= value[1] < height):
        raise ValueError(f"{name}: got {value!r}; require 0 <= u < {width} and 0 <= v < {height}")
    if length == 4 and not (0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height):
        raise ValueError(
            f"{name}: got {value!r}; use half-open [x1, y1, x2, y2] with "
            f"0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}"
        )
    return value


def format_json_path(parts) -> str:
    """Format a schema error path as an actionable field reference"""
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def extract_label_category(label) -> str | None:
    """Read the final category word while allowing prefixes such as green in green mug"""
    if not isinstance(label, str) or not label.strip():
        return None
    return label.lower().split()[-1]


def collect_object_errors(obj: dict, index: int, width: int, height: int) -> list[str]:
    """Collect independent object geometry and category errors without changing observations"""
    path = f"$.objects[{index}] ({obj.get('id', '?')})"
    errors = []
    geometry = {}
    for field, length in (("bbox_2d", 4), ("pick_point_2d", 2)):
        try:
            geometry[field] = validate_pixel(obj.get(field), f"{path}.{field}", width, height, length)
        except ValueError as error:
            errors.append(str(error))
    if len(geometry) == 2:
        x1, y1, x2, y2 = geometry["bbox_2d"]
        u, v = geometry["pick_point_2d"]
        if not (x1 <= u < x2 and y1 <= v < y2):
            errors.append(
                f"{path}.pick_point_2d = {[u, v]} lies outside bbox_2d = {[x1, y1, x2, y2]}. "
                "Select visible solid material inside the half-open box; do not clamp an unverified point"
            )

    confidence = obj.get("confidence")
    if isinstance(confidence, float) and not math.isfinite(confidence):
        errors.append(f"{path}.confidence: use a finite number from 0 to 1, not {confidence!r}")

    label = obj.get("label")
    category = extract_label_category(label)
    if category in BACKGROUND_LABELS:
        errors.append(
            f"{path}.label = {label!r} describes excluded background. "
            "Remove this background object and remove its ID from on_top_of lists; "
            "use [] when no detected supporting objects remain. Recheck any inside reference to it; "
            "do not remove actual trays, plates, or saucers"
        )
    if category in CATEGORY_KINDS and obj.get("kind") != CATEGORY_KINDS[category]:
        errors.append(
            f"{path}.kind = {obj.get('kind')!r} conflicts with label {label!r}: "
            f"a {category} has kind {CATEGORY_KINDS[category]!r} in this task. "
            "Reinspect the category rather than changing it just to pass validation"
        )
    return errors


def collect_relation_references(relations: dict, fields: tuple[str, ...]) -> list[tuple[str, str]]:
    """Expand validated containment and support references with exact field paths"""
    references = []
    for field in fields:
        if field == "inside":
            if relations[field] is not None:
                references.append((field, relations[field]))
        else:
            references.extend((f"{field}[{index}]", target) for index, target in enumerate(relations[field]))
    return references


def collect_cycle_errors(objects: dict[str, dict], paths: dict[str, str], fields: tuple[str, ...]) -> list[str]:
    """Detect directed cycles across one or both relations without recursive depth limits"""
    edges = {
        object_id: [(field, target) for field, target in collect_relation_references(obj["relations"], fields)
                    if target in objects]
        for object_id, obj in objects.items()
    }
    errors = []
    completed = set()
    for object_id in objects:
        if object_id in completed:
            continue
        chain = [object_id]
        chain_fields = []
        positions = {object_id: 0}
        pending = [(object_id, iter(edges[object_id]))]

        # Track the active path separately from completed nodes to distinguish cycles from shared parents
        while pending:
            current, remaining = pending[-1]
            edge = next(remaining, None)
            if edge is None:
                completed.add(current)
                del positions[current]
                pending.pop()
                chain.pop()
                if chain_fields:
                    chain_fields.pop()
                continue

            field, target = edge
            if target in positions:
                start = positions[target]
                cycle = chain[start:] + [target]
                cycle_fields = chain_fields[start:] + [field]
                relation_types = {relation.split("[", 1)[0] for relation in cycle_fields}
                if len(fields) > 1 and len(relation_types) == 1:
                    continue
                description = fields[0] if len(fields) == 1 else "mixed inside/on_top_of"
                assignments = ", ".join(
                    f"{paths[node]}.{relation} = {json.dumps(parent)}"
                    for node, relation, parent in zip(cycle, cycle_fields, cycle[1:])
                )
                errors.append(
                    f"Circular {description} chain: {' -> '.join(cycle)}. "
                    f"Conflicting fields: {assignments}. Reinspect the image and correct the false "
                    "references. Enclosing/supporting parents must form an acyclic hierarchy "
                    "in this tabletop scene model. Never mirror a relationship to repair it"
                )
            elif target not in completed:
                positions[target] = len(chain)
                chain.append(target)
                chain_fields.append(field)
                pending.append((target, iter(edges[target])))
    return errors


def collect_relation_errors(objects: dict[str, dict], paths: dict[str, str], width: int, height: int,
                            require_supports: bool) -> list[str]:
    """Check directed references, containment plausibility, cycles, and derived inverses"""
    errors = []
    margin = max(width, height) * CONTAINMENT_MARGIN_FRACTION
    for object_id, obj in objects.items():
        relations = obj["relations"]
        for field, target_id in collect_relation_references(relations, ("inside", "on_top_of")):
            path = f"{paths[object_id]}.{field}"
            if target_id not in objects:
                errors.append(
                    f"{path} = {json.dumps(target_id)} has no uniquely identified, structurally valid target. "
                    f"Valid IDs: {json.dumps(list(objects))}. Correct the reference or the target's fields; "
                    "do not invent an object to satisfy a reference"
                )
                continue
            target = objects[target_id]
            if target["kind"] == "human":
                errors.append(
                    f"{path} targets a human. Represent human holding with held_by_human=true "
                    "when visible, not inside/on_top_of"
                )
            if field != "inside":
                continue

            category = extract_label_category(target["label"])
            if target["kind"] != "container" and category not in RECEPTACLE_LABELS:
                errors.append(
                    f"{path} targets {target_id!r}, which is not a recognized receptacle. "
                    "inside means enclosure by a tray, mug, cup, or bowl; "
                    "resting on a plate or another solid object uses on_top_of"
                )

            # This conservative 2D plausibility gate is not proof of physical containment
            try:
                child_box = validate_pixel(obj.get("bbox_2d"), path, width, height, 4)
                parent_box = validate_pixel(target.get("bbox_2d"), path, width, height, 4)
            except ValueError:
                continue
            x1, y1, x2, y2 = child_box
            a1, b1, a2, b2 = parent_box
            if not (a1 - margin <= x1 and b1 - margin <= y1 and
                    x2 <= a2 + margin and y2 <= b2 + margin):
                errors.append(
                    f"{path} claims {object_id!r} is inside {target_id!r}, but its box {child_box} "
                    f"extends beyond the enclosing box {parent_box} by more than the {margin:.1f}px "
                    "tolerance. Check containment direction and both detections against the image. "
                    "If the object is outside the receptacle, set inside to null. If a detection boundary "
                    "is wrong, correct it from the image. Do not swap IDs or enlarge boxes just to pass validation"
                )

    for fields in (("inside",), ("on_top_of",), ("inside", "on_top_of")):
        errors.extend(collect_cycle_errors(objects, paths, fields))

    # Compare with the exact inverse; never ask perception to predict it independently
    if require_supports:
        expected = {object_id: [] for object_id in objects}
        for object_id, obj in objects.items():
            for parent in obj["relations"]["on_top_of"]:
                if parent in expected:
                    expected[parent].append(object_id)
        for object_id, obj in objects.items():
            actual = obj["relations"]["supports"]
            if sorted(actual) != sorted(expected[object_id]):
                errors.append(
                    f"{paths[object_id]}.supports = {json.dumps(actual)}; "
                    f"the inverse of on_top_of requires {json.dumps(expected[object_id])}. "
                    "Rebuild supports from validated on_top_of observations"
                )
    return errors


def validate_scene_graph(graph: dict, width: int, height: int, *, require_supports: bool = True) -> dict:
    """Collect schema and scene-consistency errors and reject the graph without guessing repairs"""
    if not is_integer(width) or not is_integer(height) or width < 1 or height < 1:
        raise ValueError("Image width and height must be positive integers")

    schema = load_scene_graph_schema() if require_supports else load_perception_schema()
    schema_errors = list(Draft202012Validator(schema).iter_errors(graph))
    errors = [f"{format_json_path(error.absolute_path)}: {error.message}" for error in schema_errors]

    # Structural failures do not hide independent geometry or relationship contradictions
    if isinstance(graph, dict):
        image = graph.get("image")
        if (not isinstance(image, dict) or
                not all(is_integer(image.get(key)) for key in ("width", "height")) or
                image != {"width": width, "height": height}):
            errors.append(f"$.image must be exactly {{'width': {width}, 'height': {height}}}")
        objects = graph.get("objects")
        if isinstance(objects, list):
            ids = [obj.get("id") for obj in objects if isinstance(obj, dict) and isinstance(obj.get("id"), str)]
            duplicates = {object_id for object_id, count in Counter(ids).items() if count > 1}
            if duplicates:
                errors.append(f"$.objects: duplicate IDs {sorted(duplicates)}; assign distinct IDs")

            valid_objects = {}
            paths = {}
            for index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                errors.extend(collect_object_errors(obj, index, width, height))

                # Geometry errors do not prevent reference checks, but malformed relation types do
                invalid_fields = {
                    error.absolute_path[2] if len(error.absolute_path) > 2 else None
                    for error in schema_errors
                    if len(error.absolute_path) >= 2 and list(error.absolute_path)[:2] == ["objects", index]
                }
                if invalid_fields - {"bbox_2d", "pick_point_2d", "confidence", "color"}:
                    continue
                object_id = obj.get("id")
                if not isinstance(object_id, str) or object_id in duplicates:
                    continue
                valid_objects[object_id] = obj
                paths[object_id] = f"$.objects[{index}] ({object_id}).relations"
            errors.extend(collect_relation_errors(valid_objects, paths, width, height, require_supports))

    if errors:
        raise ValueError("Scene graph validation failed:\n" + "\n".join(
            f"{index}. {error}" for index, error in enumerate(dict.fromkeys(errors), start=1)
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
