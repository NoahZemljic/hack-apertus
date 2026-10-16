import json
import math
from pathlib import Path

PLANNING_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "apertus_system.txt"


def prepare_planning_prompt(
    graph: dict, instruction: str, gripper_mm: float | None, *, geometry: dict | None = None,
    policy: str | None = None,
) -> str:
    """Describe the SAM-derived graph and its supplied spatial measurements."""
    opening = gripper_mm if gripper_mm is not None else 85
    geometry = derived_geometry(graph) if geometry is None else geometry
    policy = PLANNING_PROMPT_PATH.read_text().strip() if policy is None else policy.strip()
    distance_context = ""
    points = geometry.get("centroids_2d", {})
    movable = [obj["id"] for obj in graph["objects"] if obj.get("kind") == "movable" and obj["id"] in points]
    if points and movable:
        ordering = {reference: sorted((oid for oid in movable if oid != reference),
                                      key=lambda oid: (math.dist(point, points[oid]), oid))
                    for reference, point in points.items()}
        distance_context = (
            "\n\nComputed movable-object distance order from each reference (nearest first):\n"
            + json.dumps(ordering, separators=(",", ":"))
            + "\n\nFilter this ordering by the requested category and attributes. A last matching ID is farthest. "
            "Use pairwise_distances to check ties. This uses mask centroids."
        )
    blocked = [
        {"lower_object": obj["id"], "upper_objects_resting_on_it": obj["relations"]["supports"]}
        for obj in graph["objects"] if obj["relations"].get("supports")
    ]
    access_context = ""
    if blocked:
        access_context = (
            "\n\nPhysical access prerequisites (upper objects must move before lower objects):\n"
            + json.dumps(blocked, separators=(",", ":"))
            + "\n\nMove each blocking upper object once to a permitted destination, then move the lower objects. "
            "An upper object shared by two lower objects needs only one clearing transfer. "
            "Complete pick(source), place(destination) before picking the next object. Do not reverse these links."
        )
    return f"""{policy}

Scene graph:
{json.dumps(graph, separators=(",", ":"))}

Deterministically derived 2D geometry:
{json.dumps(geometry, separators=(",", ":"))}{distance_context}

Supplied field semantics:
{planning_field_semantics(graph, geometry)}{access_context}

The gripper holds one object at a time and opens at most {opening:g} mm.
Instruction: {json.dumps(instruction)}
Return the decision for this exact instruction."""


def planning_field_semantics(graph: dict, geometry: dict | None = None) -> str:
    """Keep field guidance tied to keys that are actually present in the payload."""
    fields = {key for obj in graph["objects"] for key in obj}
    descriptions = {
        "id": "the exact object identifier used in action steps",
        "label": "the observed object description; words may be separated by spaces or underscores",
        "kind": "the object's role: movable, container, surface, or human",
        "color": "the observed color; null means it is not established",
        "pick_point_2d": "a picking or destination point in image pixels, not the object's center or physical size",
        "bbox_2d": "the object's image bounds in pixels, not millimetres",
        "width_mm": "the estimated physical grasp width; a value above the gripper opening prevents picking",
        "graspable": "whether the measured width fits the gripper; an absent measurement does not establish that picking is impossible",
        "confidence": "a detector score; a low score alone does not prevent a plan",
    }
    if any("confidence" in obj and obj["confidence"] is None for obj in graph["objects"]):
        descriptions["confidence"] = (
            "a detector score, or null when the detector supplies none; "
            "unknown or low confidence alone does not prevent a plan"
        )
    guidance = [
        f"- objects[].{key}: {description}."
        for key, description in descriptions.items() if key in fields
    ]
    relation_fields = {key for obj in graph["objects"] for key in obj["relations"]}
    relation_descriptions = {
        "inside": "the containing object's ID; containment alone does not establish support",
        "on_top_of": "IDs of objects supporting this object: this object is on those objects, not the reverse; [] means no supplied support; legacy single-ID or null values mean one or no support",
        "supports": "IDs of objects resting on this object, the inverse of on_top_of; move upper objects first when access requires it and the instruction permits it",
        "held_by_human": "true forbids picking; false does not override explicit human contact in the instruction",
        "partially_occluded": "an occlusion observation, not identification of a blocker or proof of support",
        "occluded_by": "detected object IDs hiding this object; account for access in the plan",
    }
    guidance.extend(
        f"- objects[].relations.{key}: {description}."
        for key, description in relation_descriptions.items() if key in relation_fields
    )
    guidance.append("- Image-plane above/below describes camera ordering, not physical stacking.")
    source = (geometry or {}).get("support_source")
    if source == "qwen":
        guidance.append(
            "- support_source=qwen: support links are directed visual observations from a separate "
            "Qwen image request using these object IDs; [] means it reported no support, "
            "not a geometry-derived fallback."
        )
    elif source == "geometry":
        guidance.append(
            "- support_source=geometry: support links are mask-geometry hypotheses, "
            "not visual confirmation of physical stacking."
        )
    elif source == "none":
        guidance.append("- support_source=none: no support observations were supplied.")
    return "\n".join(guidance)


def derived_geometry(graph: dict) -> dict:
    """Calculate image ordering/distances using the supplied picking points."""
    objects = graph["objects"]
    points = {obj["id"]: obj["pick_point_2d"] for obj in objects}

    # Calculate spatial facts in code instead of asking Apertus to compare coordinates
    ordered = sorted(objects, key=lambda obj: (*points[obj["id"]], obj["id"]))
    distances = []
    for index, first in enumerate(objects):
        u1, v1 = points[first["id"]]
        for second in objects[index + 1:]:
            u2, v2 = points[second["id"]]
            distances.append({
                "a": first["id"],
                "b": second["id"],
                "distance_px": round(math.hypot(u2 - u1, v2 - v1), 1),
            })
    return {"coordinate_reference": "supplied picking points in image pixels; not physical distances or support evidence",
            "left_to_right": [obj["id"] for obj in ordered], "pairwise_distances": distances}
