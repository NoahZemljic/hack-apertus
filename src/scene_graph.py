"""Utilities for the grounded graph's directed physical relationships."""

from copy import deepcopy


def graph_objects_by_id(graph: dict) -> dict[str, dict]:
    return {obj["id"]: obj for obj in graph["objects"]}


def derive_support_relations(graph: dict) -> dict:
    """Derive inverse supports from on_top_of; containment supplies no support."""
    result = deepcopy(graph)
    objects = graph_objects_by_id(result)
    for obj in objects.values():
        obj["relations"]["supports"] = []
    for obj in objects.values():
        parent = obj["relations"]["on_top_of"]
        if parent is not None:
            if parent not in objects:
                raise ValueError(f"{obj['id']}: unknown supporting object {parent!r}")
            objects[parent]["relations"]["supports"].append(obj["id"])
    return result
