"""Deterministic conversion of saved mask geometry into the existing planner contract."""

from copy import deepcopy
import math
from pathlib import Path
import re

from .geometry import GeometryReasoner
from .scene_graph import derive_support_relations, graph_objects_by_id, supporting_ids
from .validation import validate_scene_graph


COLORS = {"black", "white", "red", "yellow", "green", "blue", "grey", "orange",
          "purple", "brown", "pink"}
SPATIAL_RELATIONS = {"left_of", "right_of", "above", "below", "overlaps", "touching"}


def object_properties(label: str) -> tuple[str, str | None]:
    """Classify only explicit label words; preserve the full label in the graph."""
    words = set(re.findall(r"[a-z]+", label.lower().replace("gray", "grey")))
    if words & {"hand", "arm", "human", "person"}:
        kind = "human"
    elif words & {"tray", "container", "bin", "basket"}:
        kind = "container"
    elif words & {"plate", "saucer", "coaster"}:
        kind = "surface"
    else:
        kind = "movable"
    # Orange alone denotes a fruit, not its measured color.
    colors = words & COLORS
    if words == {"orange"}:
        colors = set()
    return kind, next(iter(colors)) if len(colors) == 1 else None


def planner_box(bounds: tuple[int, int, int, int], width: int, height: int) -> list[int]:
    """Convert exclusive SAM bounds to the planner's inclusive in-image endpoints."""
    x1, y1, x2, y2 = bounds
    x2, y2 = x2 - 1, y2 - 1
    # A valid thin mask can still need a nonzero-size planner box.
    if x1 == x2:
        x1, x2 = max(0, x1 - 1), min(width - 1, x2 + 1)
    if y1 == y2:
        y1, y2 = max(0, y1 - 1), min(height - 1, y2 + 1)
    return [x1, y1, x2, y2]


def creates_cycle(objects: dict, source: str, target: str) -> bool:
    """Test the combined containment/support hierarchy before adding an edge."""
    seen = set()
    pending = [target]
    while pending:
        current = pending.pop()
        if current == source:
            return True
        if current in seen:
            continue
        seen.add(current)
        relations = objects[current]["relations"]
        pending.extend(supporting_ids(relations["on_top_of"]))
        if relations["inside"] is not None:
            pending.append(relations["inside"])
    return False


class GeometrySceneGraphBuilder:
    def __init__(self, input_path: Path, *, base_dir: Path | None = None,
                 accept_support_candidates: bool = True) -> None:
        self.reasoner = GeometryReasoner(base_dir=base_dir or input_path.resolve().parent)
        records = self.reasoner.load_detections(input_path)
        self.records = {record["scene"]: record for record in records}
        self.accept_support_candidates = accept_support_candidates

    def build(self, scene: str, width: int, height: int) -> tuple[dict, dict, dict]:
        """Return a validated graph, clean spatial context, and separate diagnostics.

        Visual support observations override geometry support hypotheses, including
        an explicitly empty observation. Legacy geometry can supply estimates when
        no visual observations exist. Neither source proves physical support.
        """
        record = self.records[scene]
        if record.get("geometry_version") != "1.0":
            raise ValueError(f"{scene}: expected geometry_version '1.0'")
        if record.get("image") != {"width": width, "height": height}:
            raise ValueError(f"{scene}: geometry image dimensions do not match RGB image")
        graph = {"image": {"width": width, "height": height}, "objects": []}
        visual = record.get("visual_supports")
        has_visual = "visual_supports" in record
        if has_visual and (not isinstance(visual, dict) or visual.get("backend") != "qwen"
                           or set(visual) != {"backend", "supports"}
                           or not isinstance(visual["supports"], list)):
            raise ValueError(f"{scene}: visual_supports must contain backend 'qwen' and a supports list")
        debug = {"scene": scene, "accept_support_candidates": self.accept_support_candidates,
                 "support_source": "qwen" if has_visual else "none",
                 "objects": [], "accepted_relations": [], "rejected_relations": []}
        centroids = {}
        mask_areas = {}
        for obj in record["objects"]:
            mask = self.reasoner.load_mask(obj, (height, width))
            kind, color = object_properties(obj["label"])
            graph["objects"].append({
                "id": obj["id"], "kind": kind, "label": obj["label"], "color": color,
                "bbox_2d": planner_box(mask.bounds, width, height),
                "pick_point_2d": list(obj["safe_pixel"]),
                "confidence": obj.get("detector_score", obj.get("dino_score", obj.get("score"))),
                "relations": {"inside": None, "on_top_of": [], "supports": [],
                              "held_by_human": False, "partially_occluded": False},
            })
            centroids[obj["id"]] = list(obj["centroid"])
            mask_areas[obj["id"]] = obj["mask_area"]
            debug["objects"].append(deepcopy(obj))

        # Validate IDs, points, boxes and types before resolving any references.
        validate_scene_graph(graph, width, height)
        objects = graph_objects_by_id(graph)
        edges = record.get("relations")
        candidates = record.get("relation_candidates")
        if not isinstance(edges, list) or not isinstance(candidates, list):
            raise ValueError(f"{scene}: geometry relations and relation_candidates must be lists")
        for edge in edges + candidates:
            if not isinstance(edge, dict) or edge.get("source") not in objects or edge.get("target") not in objects:
                raise ValueError(f"{scene}: relation references an unknown object: {edge}")
            if edge["source"] == edge["target"]:
                raise ValueError(f"{scene}: self relation: {edge}")

        # Only containers receive semantic inside links. Prefer the smallest
        # eligible container if projected enclosure supplies multiple choices.
        inside = sorted((e for e in edges if e["type"] == "inside"),
                        key=lambda e: (mask_areas[e["target"]], e["source"], e["target"]))
        support = sorted((e for e in candidates if e["type"] == "on"),
                         key=lambda e: (e["source"], e["target"]))
        for edge in inside:
            source, target = edge["source"], edge["target"]
            reason = None
            if objects[target]["kind"] != "container" or objects[source]["kind"] == "human":
                reason = "projected enclosure is not object membership in a container"
            elif objects[source]["relations"]["inside"] is not None:
                reason = "one relationship target already selected"
            elif creates_cycle(objects, source, target):
                reason = "relationship would form a cycle"
            diagnostic = {**deepcopy(edge), "evidence_source": "geometry"}
            if reason:
                debug["rejected_relations"].append({**diagnostic, "builder_reason": reason})
            else:
                objects[source]["relations"]["inside"] = target
                debug["accepted_relations"].append(diagnostic)

        # Geometry estimates remain reviewable but never override an explicit
        # visual result, even when Qwen reports no stack relationships.
        for edge in support:
            source, target = edge["source"], edge["target"]
            reason = None
            if has_visual:
                reason = "visual support observations override geometry support hypotheses"
            elif not self.accept_support_candidates:
                reason = "support candidate acceptance disabled"
            elif objects[source]["kind"] != "movable":
                reason = "support source is not movable"
            elif objects[target]["kind"] == "human":
                reason = "human contact does not establish support"
            elif objects[source]["relations"]["on_top_of"]:
                reason = "one relationship target already selected"
            elif creates_cycle(objects, source, target):
                reason = "relationship would form a cycle"
            diagnostic = {**deepcopy(edge), "evidence_source": "geometry"}
            if reason:
                debug["rejected_relations"].append({**diagnostic, "builder_reason": reason})
            else:
                objects[source]["relations"]["on_top_of"].append(target)
                debug["accepted_relations"].append(diagnostic)
                debug["support_source"] = "geometry"

        if has_visual:
            seen_supports = set()
            for edge in visual["supports"]:
                if not isinstance(edge, dict) or set(edge) != {"upper", "lower"}:
                    raise ValueError(f"{scene}: visual support must contain exactly upper and lower: {edge}")
                source, target = edge["upper"], edge["lower"]
                if (not isinstance(source, str) or not isinstance(target, str)
                        or source not in objects or target not in objects):
                    raise ValueError(f"{scene}: visual support references an unknown object: {edge}")
                if source == target:
                    raise ValueError(f"{scene}: self visual support: {edge}")
                if (source, target) in seen_supports:
                    raise ValueError(f"{scene}: duplicate visual support: {edge}")
                if objects[source]["kind"] in {"human", "container"}:
                    raise ValueError(f"{scene}: visual support upper must be a workpiece or surface: {edge}")
                if objects[target]["kind"] in {"human", "container"}:
                    raise ValueError(f"{scene}: visual support lower must be a workpiece or surface: {edge}")
                if creates_cycle(objects, source, target):
                    raise ValueError(f"{scene}: visual support would form a relationship cycle: {edge}")
                seen_supports.add((source, target))
                objects[source]["relations"]["on_top_of"].append(target)
                debug["accepted_relations"].append({"source": source, "type": "on", "target": target,
                                                    "evidence_source": "visual", "backend": visual["backend"]})

        # These flags require explicit upstream observations; overlap or contact
        # alone is not used to invent human holding or occlusion.
        for obj in record["objects"]:
            for field in ("held_by_human", "partially_occluded"):
                if field in obj:
                    if type(obj[field]) is not bool:
                        raise ValueError(f"{obj['id']}: {field} must be boolean")
                    objects[obj["id"]]["relations"][field] = obj[field]
        graph = derive_support_relations(graph)
        validate_scene_graph(graph, width, height)

        ordered = sorted(objects, key=lambda oid: (*centroids[oid], oid))
        distances = []
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                x1, y1 = centroids[first]
                x2, y2 = centroids[second]
                distances.append({"a": first, "b": second,
                                  "distance_px": round(math.hypot(x2 - x1, y2 - y1), 1)})
        spatial = {"coordinate_reference": "mask centroids; pick points are for picking only",
                   "support_source": debug["support_source"],
                   "centroids_2d": centroids, "left_to_right": ordered,
                   "pairwise_distances": distances,
                   "image_plane_relations": [
                       {"source": e["source"], "type": e["type"], "target": e["target"]}
                       for e in edges if e["type"] in SPATIAL_RELATIONS]}
        debug["input_relations"] = deepcopy(edges)
        debug["input_relation_candidates"] = deepcopy(candidates)
        if has_visual:
            debug["input_visual_supports"] = deepcopy(visual)
        return graph, spatial, debug
