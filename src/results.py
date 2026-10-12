import json
from pathlib import Path


def plan_to_result(instruction_id: str, plan: dict, graph: dict) -> dict:
    """Convert an internal plan into the challenge submission format"""
    if plan["decision"] == "ask_user":
        return {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "ask_user", "question": plan["question"]}],
        }

    if plan["decision"] == "refuse":
        return {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "refuse", "reason": plan["reason"]}],
        }

    objects = {obj["id"]: obj for obj in graph["objects"]}
    first_picked_object = objects[plan["steps"][0]["object_id"]]
    actions = [{"action": step["action"], "object": step["object_id"]} for step in plan["steps"]]

    # Ground the result from the graph rather than trusting the model to repeat coordinates
    return {
        "instruction_id": instruction_id,
        "target_pixel": first_picked_object["target_pixel"],
        "actions": actions,
    }


def write_submission(out: Path, team: str, results: list[dict]) -> None:
    """Write the final challenge submission."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"team": team, "results": results}, indent=2) + "\n")
