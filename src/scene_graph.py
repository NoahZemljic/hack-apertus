def graph_objects_by_id(graph: dict) -> dict[str, dict]:
    """Index scene objects by their unique IDs"""
    return {obj["id"]: obj for obj in graph["objects"]}
