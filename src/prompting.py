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
    return f"""{policy}

Scene graph:
{json.dumps(graph, separators=(",", ":"))}

Deterministically derived 2D geometry:
{json.dumps(geometry, separators=(",", ":"))}{distance_context}

Supplied field semantics:
{planning_field_semantics(graph)}

The gripper holds one object at a time and opens at most {opening:g} mm.
Instruction: {json.dumps(instruction)}
Return the decision for this exact instruction."""


def planning_field_semantics(graph: dict) -> str:
    """Keep field guidance tied to keys that are actually present in the payload."""
    fields = {key for obj in graph["objects"] for key in obj}
    descriptions = {
        "id": "the exact object identifier used in action steps",
        "label": "the observed object description; underscores separate words",
        "kind": "the object's role: movable, container, surface, or human",
        "color": "the observed color; null means it is not established",
        "pick_point_2d": "a picking or destination point in image pixels, not the object's center or physical size",
        "bbox_2d": "the object's image bounds in pixels, not millimetres",
        "width_mm": "the estimated physical grasp width; a value above the gripper opening prevents picking",
        "graspable": "whether the measured width fits the gripper; an absent measurement does not establish that picking is impossible",
        "confidence": "a detector score; a low score alone does not prevent a plan",
    }
    guidance = [
        f"- objects[].{key}: {description}."
        for key, description in descriptions.items() if key in fields
    ]
    relation_fields = {key for obj in graph["objects"] for key in obj["relations"]}
    relation_descriptions = {
        "inside": "the containing object's ID; containment alone does not establish support",
        "on_top_of": "the estimated supporting object's ID: this object is on that object, not the reverse; geometry is not physical confirmation",
        "supports": "IDs estimated to rest on this object; use them to plan access and prerequisites, not as automatic reasons to refuse",
        "held_by_human": "true forbids picking; false does not override explicit human contact in the instruction",
        "partially_occluded": "an occlusion observation, not identification of a blocker or proof of support",
        "occluded_by": "detected object IDs hiding this object; account for access in the plan",
    }
    guidance.extend(
        f"- objects[].relations.{key}: {description}."
        for key, description in relation_descriptions.items() if key in relation_fields
    )
    guidance.append("- Image-plane above/below describes camera ordering, not physical stacking.")
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


def prepare_scene_discovery_prompt(hints: list[str]) -> str:
    """Build the Qwen prompt for visible-object discovery."""
    return f"""
You are the visual perception module for a tabletop robot.

Inspect the entire image and identify the object categories that are actually visible in the robot workspace.

To do so you will be provided with a list of possible objects present in the image:

{json.dumps(hints, ensure_ascii=False)}

Important rules:
-Try to describe the objects by their color and their category: <color> <object category>
- The hints are clues only. Only return objects that are actually visible in the image.
- The hints are not exhaustive. Search the entire image carefully for additional visible objects, including small, partially occluded, stacked, or peripheral objects.
- Hints may contain task relations, synonyms, or absent objects. Describe the visible physical category and color; do not copy relational phrases from a hint into an object label.
- Return object labels in singular form. Multiple visible instances of the same category require only one label.
- Use normal lowercase words separated by spaces. Never use snake_case, underscores, IDs, numbering, or plural instance names.
- Do not use generic referential terms that do not identify a visual object category.
- Do not add unnecessary visual attributes to a label when a matching hint already provides an adequate name.
- Do not use phrases such as "object on top of", "under the", "the one", or "part next to" as categories. Resolve the visible object type instead.
- A mechanical part with repeated teeth around its rim is a gear, even with a solid raised hub rather than a hole. Check carefully for small or partly covered gears and cubes.
- Inspect human hands and arms as separate visible categories, including those gripping an object.
- Inspect container rims and boundaries at the image edges. A dark or black tray may fill most of the image and contain the smaller objects; identify it when its physical rim or tray shape is visible. Distinguish it from the background table, and include it even if partially cropped.
- Keep grey and black tray categories separate when both are visibly present; do not infer an absent tray from an instruction.
- For visible objects that do not match any hint, use a short, specific, detector-friendly object name based only on what is visually observable.
- Include visually distinguishable containers and other relevant physical objects, but do not include the table or workbench itself.
- Return each object label only once.
"""

def prepare_verification_prompt(
    hints: list[str],
    objects: list[str],
) -> str:
    return f"""
Reinspect the entire image and if necessary correct this proposed list of visible objects:

{json.dumps(objects, ensure_ascii=False)}

Check for:
- visible object types that are missing
- incorrect object names
- objects that are not actually visible

The list may already contain every visible category. Hints may name absent objects; verify against the image:
{json.dumps(hints, ensure_ascii=False)}

Resolve relational phrases into physical categories. Keep colors tied to the actual object.
Search for small and partly covered gears, cubes, hands and arms throughout the image.
Check the dark surface beneath the objects for a visible tray rim or boundary. A large black tray must not be omitted as background when its tray shape is visible; do not invent one when it is only a table.
Do not remove a real physical category merely because it is not in the hints.

Use short detector-friendly phrases such as the ones given in the list.
Do not use plural for any of the objects, always return object names in their singular form.

Return the complete corrected list.
"""
