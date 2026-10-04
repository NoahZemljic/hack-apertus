from copy import deepcopy

def graph_objects_by_id(graph: dict) -> dict[str, dict]:
    """Index scene objects by their unique IDs"""
    return {obj["id"]: obj for obj in graph["objects"]}


def derive_support_relations(graph: dict) -> dict:
    """Build inverse support lists from validated observations without mutating them"""
    result = deepcopy(graph)
    objects = graph_objects_by_id(result)
    for obj in objects.values():
        obj["relations"]["supports"] = []

    # Containment alone never establishes physical support
    for obj in objects.values():
        for supporting_id in obj["relations"]["on_top_of"]:
            objects[supporting_id]["relations"]["supports"].append(obj["id"])
    return result
