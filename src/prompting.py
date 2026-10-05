import json
import math
from pathlib import Path

from .schemas import load_perception_schema

PERCEPTION_RELATION_RULES = """Relationship semantics:
- Identify every object and its boundaries before assigning relationships between detections
- inside is the ID of the immediate enclosing receptacle, or null; it points from contents to receptacle
- A tray has an enclosing rim, side walls, and a recessed interior. Include distinct flat work mats used as placement areas, even if their borders are clipped by the image. They are surface objects with label "work mat", not enclosing trays and not excluded background. Detect genuine raised-wall trays of any color
- An object beside a tray is not inside it. Verify that it occupies the receptacle's interior, not merely the same workspace. An empty tray must remain empty in the graph
- A movable mug may contain an object. A large surrounding tray is not inside a smaller mug
- on_top_of is ALWAYS a list of all detected objects directly supporting this object from below. Use [] for background support or when no detected support is established, never null or a single string
- If object_a rests directly across support_b and support_c, object_a.on_top_of = ["support_b", "support_c"]. Include both only when both contacts are evidenced, not merely because their boxes overlap
- Include only direct supports, not objects further below in a stack. Do not include the supported object itself or duplicate IDs
- Containment and support are independent. If part_a is enclosed by receptacle_b and rests on its base, part_a.inside = "receptacle_b" and part_a.on_top_of = ["receptacle_b"], never the reverse
- Neither relation is symmetric. Following any combination of inside and on_top_of must never return to the starting object
- Box overlap alone proves neither enclosure nor supporting contact. Use visible boundaries, occlusion ordering, and contact evidence
- Output only inside, on_top_of, held_by_human, and partially_occluded in relations. Do not output supports; code derives the inverse from every on_top_of entry

Correction procedure:
Reinspect each flagged field using the original image. If an object is outside the referenced
receptacle, set inside to null. If a claimed supporting contact is absent, remove that ID from
on_top_of; if no detected supports remain, use []. Correct incorrect detection boundaries from
the image rather than enlarging boxes to satisfy a relationship. Preserve a field only if it is
actually correct. Recheck all affected objects and return the complete corrected graph, not the
rejected graph unchanged. Do not invent contacts or remove true contacts to silence validation."""

def prepare_perception_prompt(width: int, height: int) -> str:
    """Build the perception prompt with object cues, directed relations, and pixel conventions"""
    schema = json.dumps(load_perception_schema(width, height), separators=(",", ":"))
    return f"""Analyze this overhead robot-workspace image and construct a complete 2D scene graph.

The original image is {width} x {height} pixels. Coordinates are [u, v], where u increases rightward and v increases downward. Return all coordinates in this original image space.

Bounding boxes use half-open edges: [x1, y1, x2, y2], with 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}. The right and bottom edges may equal the image dimensions. Pick points are actual integer pixels: 0 <= u < {width}, 0 <= v < {height}, and x1 <= u < x2, y1 <= v < y2. Never use width or height itself as a pick-point index.

Object identification:
This workspace may contain mechanical parts, household objects, fruit, trays, and human hands. These are possible categories, not a required inventory. Detect only visible objects and do not invent absent ones.
- Gears have regularly spaced teeth and may have solid raised hubs, central bores, or neither visible. A central hole is NOT required. Identify mechanical shape rather than color: a toothed part with a solid hub is a gear, not a sun, star, or flower
- Elongated flat parts with mounting holes, large bores, or engraved arrows are industrial metal parts, not ceramic saucers. Bent or folded sheet-like parts may be brackets. Use the conservative label "industrial component" for assembled components, such as white/blue assemblies, when their exact function is unclear
- Reserve "plate" and "saucer" for ceramic dishware. Use "metal part" for industrial plate-like components and "bracket" for identifiable brackets. Printed patterns on dishware, holes, shadows, and highlights are not separate objects
- Detect mugs, cube-like blocks, and fruit such as bananas or mandarins when present. Use lowercase category labels such as "mug", "cube", or "gear", and record color separately. Include distinctive interior colors as well as exterior colors for multicolored objects (for example "white and green"). Use descriptive compound colors such as "red-orange" when a hue lies between basic color names
- Classify trays as container, ceramic plates/saucers and bounded work mats as surface, hands as human, and graspable objects including mugs, gears, metal parts, brackets, and industrial components as movable
Record every separate physical object, including repeated and partially visible instances. Assign unique snake_case IDs numbered left to right within each category. Exclude the background table and floor. Include bounded work mats as placement destinations.

{PERCEPTION_RELATION_RULES}

Pick points and occlusion:
Choose pick_point_2d on visible solid material belonging to the object, not automatically at the bounding-box centre. For perforated metal parts avoid ALL holes, including central bores and mounting holes. For gears with solid hubs, the visible hub top is a candidate. For mugs avoid the open cavity and any human hand; select visible material such as an unobstructed handle. Never choose an occluding object's pixels. For trays and ceramic surfaces choose an unobstructed representative interior point; for hands choose a representative visible point, not a grasp instruction.
Keep partially visible objects as separate detections. Mark the object hidden by another object as partially_occluded=true; do not merge overlapping objects. When a hand visibly grips an object, set that object's held_by_human=true. A nearby hand alone does not establish holding. Booleans describe visible evidence, not arbitrary defaults.

Output:
Report observations only. Do not plan actions or infer an instruction. If no in-scope objects are visible, return objects: []. Include every required field. inside must be an existing object ID or null; on_top_of must be a list of distinct existing IDs or []. Never use "unknown" as a relation target. A graph passing validation is not proof of a safe grasp. Return exactly one JSON object matching this JSON Schema:
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
In this challenge, instructions may call the black work mat the "black tray". Resolve that phrase to a detected black work mat when present. It is a placement surface, so objects on it use on_top_of rather than inside. Do not invent a missing destination.

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
