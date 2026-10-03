import json
from pathlib import Path

from .scene_graph import graph_objects_by_id


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

    objects = graph_objects_by_id(graph)
    first_picked_object = objects[plan["steps"][0]["object_id"]]
    actions = [{"action": step["action"], "object": step["object_id"]} for step in plan["steps"]]

    # Ground the result from the graph rather than trusting the model to repeat coordinates
    return {
        "instruction_id": instruction_id,
        "target_pixel": first_picked_object["pick_point_2d"],
        "actions": actions,
    }


def write_submission_files(out: Path, team: str, results: list[dict], raw: list[dict]) -> Path:
    """Write the submission and raw response log, returning the log's path"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"team": team, "results": results}, indent=2) + "\n")

    # Keep the diagnostic log separate from the evaluator's submission contract
    raw_path = out.with_suffix(".raw.jsonl")
    raw_path.write_text("".join(json.dumps(entry) + "\n" for entry in raw))
    return raw_path
