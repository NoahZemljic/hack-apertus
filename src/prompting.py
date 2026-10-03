import json
import math
from pathlib import Path

from .schemas import load_scene_graph_schema

PERCEPTION_RELATION_RULES = """Relationship semantics:
- A.on_top_of = B means A rests directly on B; B.supports must then contain A
- B.supports containing A means the same contact, not the reverse contact
- Support is directional, never symmetric, and must not form cycles
- A.inside = B means containment only; it does not establish physical support
- An object may be both inside a tray and resting directly on its base
- Only claim contacts supported by the image; use null for an unrepresented support such as the table
When repairing a contact, update both on_top_of and supports consistently. Recheck all affected objects.
Do not remove true contacts or invent contacts just to pass validation. Preserve unrelated correct detections,
IDs, boxes, and pick points unless the image or a reported error gives a reason to change them."""

def prepare_perception_prompt(width: int, height: int) -> str:
    """Builds the Qwen prompt for grounded scene-graph extraction"""
    schema = json.dumps(load_scene_graph_schema(), separators=(",", ":"))
    return f"""Analyze this overhead robot-workspace image and construct a complete 2D scene graph.

The original image is {width} x {height} pixels. Coordinates are [u, v], where u increases rightward and v increases downward. Return all coordinates in this original image space.

Detect every movable object, grey or black tray, plate, saucer, mug, and visible human hand. Classify trays as container, plates and saucers as surface, hands as human, and graspable objects including mugs as movable. For repeated objects, create unique snake_case IDs numbered from left to right. A pick_point_2d must lie on a visible solid part of the object's top surface, never in a gear hole, empty bounding-box area, or an occluding object. For containers and surfaces, choose a representative interior point. Use on_top_of and supports only for contact between detected objects; omit the table relation by using null. Use inside for membership in a tray or other container.

{PERCEPTION_RELATION_RULES}

Report observations only. Do not plan actions, infer an instruction, or invent absent objects. Use null, [], false, or "unknown" rather than omitting a required field. Return exactly one JSON object matching this JSON Schema:
{schema}"""


def prepare_planning_prompt(graph: dict, instruction: str, gripper_mm: float | None) -> str:
    """Build the Apertus prompt from an instruction and validated scene graph"""
    opening = gripper_mm if gripper_mm is not None else 85
    geometry = derived_geometry(graph)
    return f"""You are the deliberative planner for a UR5 robot arm above a table.

Instruction: {json.dumps(instruction)}

Validated scene graph:
{json.dumps(graph, separators=(",", ":"))}

Deterministically derived 2D geometry:
{json.dumps(geometry, separators=(",", ":"))}

The gripper holds one object at a time and opens at most {opening:g} mm. A plate and saucer are wider than the gripper and cannot be picked. Trays are placement destinations, not movable objects.

Choose exactly one decision:
- plan when the instruction has one safe, physically possible interpretation
- ask_user when multiple visible objects satisfy an underspecified reference
- refuse when an object is absent, the operation is impossible, or it is unsafe

Unsafe operations include throwing, dropping, pushing an object off the table, manipulating an object held by a human, or pulling a supporting object out so another object falls. To access a covered or supported object, first move objects resting on it. For each plan, list every pick and place in physical execution order. A place step's object_id is the destination object. Use only IDs from the scene graph. The first step must be pick.

For plan, question and reason must be null. For ask_user, steps must be empty, question must be specific, and reason must be null. For refuse, steps must be empty, question must be null, and reason must be specific. Return exactly one JSON object."""


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
    """Calculate exact left-to-right ordering and pairwise pixel distances"""
    objects = graph["objects"]

    # Calculate spatial facts in code instead of asking Apertus to compare coordinates
    ordered = sorted(objects, key=lambda obj: (obj["pick_point_2d"][0], obj["pick_point_2d"][1]))
    distances = []
    for index, first in enumerate(objects):
        u1, v1 = first["pick_point_2d"]
        for second in objects[index + 1:]:
            u2, v2 = second["pick_point_2d"]
            distances.append({
                "a": first["id"],
                "b": second["id"],
                "distance_px": round(math.hypot(u2 - u1, v2 - v1), 1),
            })
    return {"left_to_right": [obj["id"] for obj in ordered], "pairwise_distances": distances}
