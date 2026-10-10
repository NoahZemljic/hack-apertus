"""Serialize computed facts into the compact graph consumed by Apertus."""


def create_scene_graph(objects: list[dict], relations: list[dict]) -> dict:
    fields = ("id", "label", "target_pixel", "width_mm", "graspable")
    return {
        "objects": [{key: obj[key] for key in fields if key in obj} for obj in objects],
        "relations": relations,
    }
