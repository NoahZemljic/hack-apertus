import json
import math
from pathlib import Path

from .schemas import load_perception_schema

PLANNING_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "apertus_system.txt"

PERCEPTION_RELATION_RULES = """Relationship semantics:
- A.on_top_of = B means A rests directly on B, not that B rests on A
- Output only inside, on_top_of, held_by_human, and partially_occluded in relations
- Do not output supports; code derives that inverse relationship from on_top_of
- Support is directional, never symmetric, and must not form cycles
- A.inside = B means A is contained in B, not that B is inside A; containment must not form cycles
- Containment alone does not establish physical support
- An object may be both inside a tray and resting directly on its base
- Only claim contacts supported by the image; use null for an unrepresented support such as the table
When repairing a relationship, correct inside or on_top_of using the image. Recheck all affected objects.
Do not remove true contacts or invent contacts. Preserve unrelated correct detections,
IDs, boxes, and pick points unless the image or a reported error gives a reason to change them."""

def prepare_perception_prompt(width: int, height: int) -> str:
    """Builds the Qwen prompt for grounded scene-graph extraction"""
    schema = json.dumps(load_perception_schema(), separators=(",", ":"))
    return f"""Analyze this overhead robot-workspace image and construct a complete 2D scene graph.

The original image is {width} x {height} pixels. Coordinates are [u, v], where u increases rightward and v increases downward. Return all coordinates in this original image space.

Detect every movable object, grey or black tray, plate, saucer, mug, and visible human hand. Classify trays as container, plates and saucers as surface, hands as human, and graspable objects including mugs as movable. For repeated objects, create unique snake_case IDs numbered from left to right. A pick_point_2d must lie on a visible solid part of the object's top surface, never in a gear hole, empty bounding-box area, or an occluding object. For containers and surfaces, choose a representative interior point. Use on_top_of only for direct contact with a supporting detected object; omit the table relation by using null. Use inside for membership in a tray or other container.

Object identification: this is a workspace containing physical parts, not a collection of pictograms. A circular mechanical part with regularly spaced teeth around its rim is a gear (cogwheel), not a sun, star, or flower. Gears may have a solid raised cylindrical hub, a central bore, or neither visible; a central hole is NOT required. A toothed part with a solid raised hub is still a gear. Identify it by its mechanical shape, not color alone. When these mechanical features are visible, use label "gear", kind "movable", and IDs such as "gear_1"; record its color separately. Do not assume a gear is present without visual evidence. Place its pick_point_2d on visible solid material, such as the top of a solid hub; avoid a central hole only if one is actually present.

{PERCEPTION_RELATION_RULES}

Report observations only. Do not plan actions, infer an instruction, or invent absent objects. Use null, [], false, or "unknown" rather than omitting a required field. Return exactly one JSON object matching this JSON Schema:
{schema}"""


def prepare_planning_prompt(
    graph: dict, instruction: str, gripper_mm: float | None, *, geometry: dict | None = None,
    policy: str | None = None,
) -> str:
    """Describe only fields supplied by the current or saved grounded graph."""
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
        "target_pixel": "a picking or destination point in image pixels, not physical size",
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
    if isinstance(graph.get("relations"), list):
        guidance.append(
            "- relations is a list of subject/relation/object/confidence entries. "
            "image_left_of, image_right_of, image_up_from, and image_down_from describe image-plane directions from subject to object; "
            "their inverses are implicit. inside describes containment, not physical support. "
            "Relation confidence measures geometric evidence, not action safety."
        )
    relation_fields = {key for obj in graph["objects"] for key in obj.get("relations", {})}
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


def prepare_perception_messages(image_path: Path, width: int, height: int) -> list[dict]:
    """Attach a scene image to its perception prompt in chat-message format"""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path.resolve())},
            {"type": "text", "text": prepare_perception_prompt(width, height)},
        ],
    }]


def append_validation_feedback(
    messages: list[dict], response: str, error: ValueError, guidance: str = ""
) -> None:
    """Append a failed response and correction request to an existing conversation"""
    # Preserve the conversation so either model can repair its previous response
    messages.extend([
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": (
                f"The previous response failed validation:\n{error}\n\n"
                f"{guidance}\n\n"
                "Resolve every listed issue using the original input as evidence. "
                "Return the complete corrected JSON object, not a patch or explanation. "
                "Keep all required fields and recheck consistency across the entire response"
            ),
        },
    ])


def derived_geometry(graph: dict) -> dict:
    """Calculate image ordering/distances using the supplied picking points."""
    objects = graph["objects"]
    points = {
        obj["id"]: obj["target_pixel"] if "target_pixel" in obj else obj["pick_point_2d"]
        for obj in objects
    }

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
