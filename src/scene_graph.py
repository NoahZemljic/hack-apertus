"""Utilities for the grounded graph's directed physical relationships."""

from copy import deepcopy


def graph_objects_by_id(graph: dict) -> dict[str, dict]:
    return {obj["id"]: obj for obj in graph["objects"]}


def supporting_ids(value: str | list[str] | None) -> list[str]:
    """Read current support lists and legacy single-parent scene graphs."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(oid, str) for oid in value):
        return list(value)
    raise ValueError("on_top_of must be a list of supporting object IDs, a legacy ID, or null")


def derive_support_relations(graph: dict) -> dict:
    """Normalize support lists and derive inverses without inferring containment."""
    result = deepcopy(graph)
    objects = graph_objects_by_id(result)
    for obj in objects.values():
        obj["relations"]["supports"] = []
    for obj in objects.values():
        parents = supporting_ids(obj["relations"]["on_top_of"])
        obj["relations"]["on_top_of"] = parents
        for parent in parents:
            if parent not in objects:
                raise ValueError(f"{obj['id']}: unknown supporting object {parent!r}")
            objects[parent]["relations"]["supports"].append(obj["id"])
    return result
