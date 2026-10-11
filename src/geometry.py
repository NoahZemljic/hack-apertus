"""Deterministic image geometry from saved SAM masks, with no model inference.

Run ``python -m src.geometry``. Pixel relations describe the image plane.
Physical support (on/under) is deliberately exported as a reviewable hypothesis,
never as the scene graph's validated ``on_top_of`` relation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    """All inference thresholds; pixel thresholds scale with image resolution."""

    reference_image_side: float = 2048.0
    direction_min_pixels: float = 8.0
    direction_size_fraction: float = 0.20
    direction_projection_overlap: float = 0.25
    direction_max_normalized_distance: float = 3.0
    direction_neighbors: int = 1
    direction_nested_fraction: float = 0.80
    overlap_min_pixels: float = 16.0
    overlap_min_fraction: float = 0.02
    inside_mask_fraction: float = 0.95
    inside_enclosure_fraction: float = 0.98
    inside_min_area_ratio: float = 1.25
    touch_tolerance_pixels: float = 3.0
    touch_min_pixels: float = 1.0
    touch_min_boundary_fraction: float = 0.03
    touch_max_overlap_fraction: float = 0.10
    stack_side_min_area_ratio: float = 0.25
    stack_side_max_area_ratio: float = 4.0
    stack_side_min_projection_overlap: float = 0.65
    stack_side_max_horizontal_offset: float = 0.25
    stack_side_min_contact_fraction: float = 0.15
    stack_side_max_overlap_fraction: float = 0.35
    stack_top_min_area_ratio: float = 0.10
    stack_top_max_area_ratio: float = 0.65
    stack_top_min_mask_coverage: float = 0.80
    stack_top_min_outer_contact_fraction: float = 0.15
    stack_top_min_vertical_offset: float = 0.12
    stack_top_max_vertical_offset: float = 0.50
    stack_top_max_horizontal_offset: float = 0.30
    stack_top_max_aspect_ratio: float = 1.60
    stack_top_min_solidity: float = 0.70
    stack_ambiguity_fraction: float = 0.10
    metadata_centroid_tolerance: float = 0.01

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if self.reference_image_side <= 0:
            raise ValueError("reference_image_side must be positive")
        if type(self.direction_neighbors) is not int or self.direction_neighbors < 1:
            raise ValueError("direction_neighbors must be a positive integer")
        fractions = (
            "direction_projection_overlap", "direction_nested_fraction",
            "overlap_min_fraction", "inside_mask_fraction", "inside_enclosure_fraction",
            "touch_min_boundary_fraction", "touch_max_overlap_fraction",
            "stack_side_min_projection_overlap", "stack_side_min_contact_fraction",
            "stack_side_max_overlap_fraction", "stack_top_min_mask_coverage",
            "stack_top_min_outer_contact_fraction", "stack_top_min_solidity",
            "stack_ambiguity_fraction",
        )
        for name in fractions:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.inside_min_area_ratio <= 1:
            raise ValueError("inside_min_area_ratio must exceed 1 to prevent mutual containment")
        for prefix, suffix in (("stack_side", "area_ratio"), ("stack_top", "area_ratio"),
                               ("stack_top", "vertical_offset")):
            if getattr(self, f"{prefix}_min_{suffix}") > getattr(self, f"{prefix}_max_{suffix}"):
                raise ValueError(f"{prefix} minimum {suffix} exceeds maximum")


PIXEL_RELATIONS = ("left_of", "right_of", "above", "below", "overlaps", "inside", "contains", "touching")
INVERSES = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above",
            "inside": "contains", "contains": "inside", "on": "under", "under": "on"}
REQUIRED_FIELDS = ("id", "label", "aliases", "box", "centroid", "safe_pixel", "mask_area", "mask_path")


@dataclass
class MaskGeometry:
    record: dict[str, Any]
    # Arrays are cropped to tight mask bounds, and never serialized.
    mask: np.ndarray
    enclosed: np.ndarray
    hull: np.ndarray
    outer_boundary: np.ndarray
    bounds: tuple[int, int, int, int]
    boundary_x: np.ndarray
    boundary_y: np.ndarray
    top: np.ndarray
    bottom: np.ndarray

    @property
    def id(self) -> str:
        return self.record["id"]

    @property
    def area(self) -> int:
        return self.record["mask_area"]

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]

    @property
    def enclosed_area(self) -> int:
        return int(np.count_nonzero(self.enclosed))

    @property
    def solidity(self) -> float:
        return self.area / int(np.count_nonzero(self.hull))

    @property
    def aspect_ratio(self) -> float:
        return max(self.width, self.height) / min(self.width, self.height)


def _component(value: Any, description: str) -> str:
    if not isinstance(value, str) or value in ("", ".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {description}: expected a filename component")
    return value


def _intersection(a: MaskGeometry, b: MaskGeometry, region_b: np.ndarray | None = None) -> int:
    x1, y1 = max(a.bounds[0], b.bounds[0]), max(a.bounds[1], b.bounds[1])
    x2, y2 = min(a.bounds[2], b.bounds[2]), min(a.bounds[3], b.bounds[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    region_b = b.mask if region_b is None else region_b
    am = a.mask[y1-a.bounds[1]:y2-a.bounds[1], x1-a.bounds[0]:x2-a.bounds[0]]
    bm = region_b[y1-b.bounds[1]:y2-b.bounds[1], x1-b.bounds[0]:x2-b.bounds[0]]
    return int(np.count_nonzero(am & bm))


def _projection_overlap(a: MaskGeometry, b: MaskGeometry, axis: int) -> float:
    # Project actual foreground pixels, including gaps in disconnected masks.
    pa, pb = np.any(a.mask, axis=axis), np.any(b.mask, axis=axis)
    coordinate = 0 if axis == 0 else 1
    start = max(a.bounds[coordinate], b.bounds[coordinate])
    end = min(a.bounds[coordinate + 2], b.bounds[coordinate + 2])
    if end <= start:
        return 0.0
    shared = np.count_nonzero(pa[start-a.bounds[coordinate]:end-a.bounds[coordinate]] &
                              pb[start-b.bounds[coordinate]:end-b.bounds[coordinate]])
    return float(shared / min(np.count_nonzero(pa), np.count_nonzero(pb)))


class GeometryReasoner:
    def __init__(self, config: GeometryConfig | None = None, *, base_dir: Path = Path("."),
                 dataset_dir: Path | None = None) -> None:
        self.config = config or GeometryConfig()
        self.base_dir = Path(base_dir)
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else None

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.base_dir / path

    def load_detections(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Segmentation results must be a list of scene records")
        seen: set[str] = set()
        for scene in data:
            if not isinstance(scene, dict):
                raise ValueError("Each scene must be an object")
            name = _component(scene.get("scene"), "scene name")
            if name in seen:
                raise ValueError(f"Duplicate scene: {name}")
            seen.add(name)
            if not isinstance(scene.get("objects"), list):
                raise ValueError(f"{name}: missing objects list")
            ids: set[str] = set()
            for obj in scene["objects"]:
                if not isinstance(obj, dict) or any(key not in obj for key in REQUIRED_FIELDS):
                    raise ValueError(f"{name}: missing segmented object metadata: {REQUIRED_FIELDS}")
                oid = _component(obj["id"], "object ID")
                if oid in ids:
                    raise ValueError(f"{name}: duplicate object ID {oid}")
                ids.add(oid)
        return data

    def load_mask(self, record: dict[str, Any], image_shape: tuple[int, int]) -> MaskGeometry:
        path = self.resolve(record["mask_path"])
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"{record['id']}: cannot load mask {path}")
        if raw.ndim != 2 or raw.shape != image_shape or not np.isin(raw, [0, 1, 255]).all():
            raise ValueError(f"{record['id']}: expected a binary mask matching image shape {image_shape}")
        mask = (raw != 0).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if not xs.size:
            raise ValueError(f"{record['id']}: empty mask")
        if type(record["mask_area"]) is not int or xs.size != record["mask_area"]:
            raise ValueError(f"{record['id']}: mask_area disagrees with PNG")
        centroid = np.asarray(record["centroid"], dtype=float)
        if (centroid.shape != (2,) or not np.isfinite(centroid).all()
            or not (0 <= centroid[0] < image_shape[1] and 0 <= centroid[1] < image_shape[0])
            or not np.allclose(
            centroid, [xs.mean(), ys.mean()], atol=self.config.metadata_centroid_tolerance, rtol=0
        )):
            raise ValueError(f"{record['id']}: centroid disagrees with PNG")
        point = record["safe_pixel"]
        if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int for v in point):
            raise ValueError(f"{record['id']}: safe_pixel must be integer [x, y]")
        x, y = point
        if not (0 <= x < image_shape[1] and 0 <= y < image_shape[0] and mask[y, x]):
            raise ValueError(f"{record['id']}: safe_pixel is outside mask")
        box = np.asarray(record["box"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{record['id']}: invalid DINO box")
        if not isinstance(record["label"], str) or not isinstance(record["aliases"], list) or any(
            not isinstance(alias, str) for alias in record["aliases"]
        ):
            raise ValueError(f"{record['id']}: invalid label/aliases")

        bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        x1, y1, x2, y2 = bounds
        if "mask_box" in record and record["mask_box"] != list(bounds):
            raise ValueError(f"{record['id']}: mask_box disagrees with PNG")
        cropped = mask[y1:y2, x1:x2].copy()
        contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enclosed = np.zeros_like(cropped)
        cv2.drawContours(enclosed, contours, -1, 1, cv2.FILLED)
        boundary = np.zeros_like(cropped)
        cv2.drawContours(boundary, contours, -1, 1, 1)
        hull = np.zeros_like(cropped)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(cropped)), 1)
        by, bx = np.nonzero(boundary)
        occupied = np.any(cropped, axis=0)
        top = np.where(occupied, np.argmax(cropped, axis=0) + y1, np.nan)
        bottom = np.where(occupied, cropped.shape[0] - 1 - np.argmax(cropped[::-1], axis=0) + y1, np.nan)
        return MaskGeometry(deepcopy(record), cropped, enclosed, hull, boundary, bounds,
                            bx + x1, by + y1, top, bottom)

    def measure_boundaries(self, objects: list[MaskGeometry], image_shape: tuple[int, int],
                           tolerance: float) -> dict[tuple[str, str], dict[str, Any]]:
        """One exact distance transform per object, reused for all other objects.

        External contours exclude occlusion holes, which otherwise look like
        physical contact with an object in the hole. Peak distance-map memory is
        one image, rather than one image per object/pair.
        """
        measurements: dict[tuple[str, str], dict[str, Any]] = {}
        for target in objects:
            background = np.ones(image_shape, np.uint8)
            background[target.boundary_y, target.boundary_x] = 0
            distances = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            for source in objects:
                if source.id == target.id:
                    continue
                values = distances[source.boundary_y, source.boundary_x]
                close = int(np.count_nonzero(values <= tolerance))
                measurements[(source.id, target.id)] = {
                    "min_distance": float(values.min()), "median_distance": float(np.median(values)),
                    "contact_pixels": close, "boundary_pixels": int(values.size),
                    "contact_fraction": close / int(values.size),
                }
        return measurements

    @staticmethod
    def measure_support_profile(a: MaskGeometry, b: MaskGeometry, tolerance: float) -> dict[str, Any]:
        start, end = max(a.bounds[0], b.bounds[0]), min(a.bounds[2], b.bounds[2])
        gaps = np.array([], dtype=float)
        if end > start:
            gaps = b.top[start-b.bounds[0]:end-b.bounds[0]] - a.bottom[start-a.bounds[0]:end-a.bounds[0]] - 1
            gaps = gaps[np.isfinite(gaps)]
        contact = int(np.count_nonzero(np.abs(gaps) <= tolerance))
        return {"shared_columns": int(gaps.size), "contact_columns": contact,
                "contact_fraction": contact / min(a.width, b.width),
                "median_vertical_gap": float(np.median(gaps)) if gaps.size else None}

    def measure_pair(self, a: MaskGeometry, b: MaskGeometry, *, scale: float,
                     boundaries: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
        c = self.config
        dx = float(b.record["centroid"][0] - a.record["centroid"][0])
        dy = float(b.record["centroid"][1] - a.record["centroid"][1])
        mean_width, mean_height = (a.width + b.width) / 2, (a.height + b.height) / 2
        inter = _intersection(a, b)
        union = a.area + b.area - inter
        tolerance = max(c.touch_min_pixels, c.touch_tolerance_pixels * scale)
        return {
            "a": a.id, "b": b.id, "centroid_dx": dx, "centroid_dy": dy,
            "centroid_distance": math.hypot(dx, dy),
            "normalized_dx": dx / mean_width, "normalized_dy": dy / mean_height,
            "horizontal_dead_zone": max(c.direction_min_pixels * scale, c.direction_size_fraction * mean_width),
            "vertical_dead_zone": max(c.direction_min_pixels * scale, c.direction_size_fraction * mean_height),
            "x_projection_overlap": _projection_overlap(a, b, 0),
            "y_projection_overlap": _projection_overlap(a, b, 1),
            "intersection_pixels": inter, "union_pixels": union, "mask_iou": inter / union,
            "fraction_a_in_b": inter / a.area, "fraction_b_in_a": inter / b.area,
            "fraction_a_in_enclosed_b": _intersection(a, b, b.enclosed) / a.area,
            "fraction_b_in_enclosed_a": _intersection(b, a, a.enclosed) / b.area,
            "fraction_a_in_hull_b": _intersection(a, b, b.hull) / a.area,
            "fraction_b_in_hull_a": _intersection(b, a, a.hull) / b.area,
            "area_ratio_a_to_b": a.area / b.area,
            "enclosed_area_ratio_b_to_a": b.enclosed_area / a.enclosed_area,
            "outer_boundary_a_to_b": boundaries[(a.id, b.id)],
            "outer_boundary_b_to_a": boundaries[(b.id, a.id)],
            "support_profile_a_on_b": self.measure_support_profile(a, b, tolerance),
            "support_profile_b_on_a": self.measure_support_profile(b, a, tolerance),
        }

    def _inside_method(self, fraction: float, enclosure_fraction: float, region_ratio: float) -> str | None:
        c = self.config
        if region_ratio < c.inside_min_area_ratio:
            return None
        if fraction >= c.inside_mask_fraction:
            return "actual_mask"
        if enclosure_fraction >= c.inside_enclosure_fraction:
            return "enclosed_mask_region"
        return None

    def _stack_evidence(self, a: MaskGeometry, b: MaskGeometry, m: dict[str, Any],
                        a_is_first: bool, nested: bool) -> dict[str, Any]:
        """Evaluate two independent patterns; neither proves physical depth/order."""
        c = self.config
        direction = 1 if a_is_first else -1
        suffix = "a_on_b" if a_is_first else "b_on_a"
        coverage = m["fraction_a_in_b" if a_is_first else "fraction_b_in_a"]
        boundary = m["outer_boundary_a_to_b" if a_is_first else "outer_boundary_b_to_a"]
        max_overlap = max(m["fraction_a_in_b"], m["fraction_b_in_a"])
        ratio = a.area / b.area
        dx = -direction * m["centroid_dx"] / b.width
        dy = -direction * m["centroid_dy"] / b.height
        profile = m[f"support_profile_{suffix}"]
        side_checks = {
            "not_nested": not nested,
            "a_above_b": direction * m["centroid_dy"] > m["vertical_dead_zone"],
            "area_ratio": c.stack_side_min_area_ratio <= ratio <= c.stack_side_max_area_ratio,
            "horizontal_alignment": abs(m["normalized_dx"]) <= c.stack_side_max_horizontal_offset,
            "shared_projection": m["x_projection_overlap"] >= c.stack_side_min_projection_overlap,
            "opposing_mask_edges": profile["contact_fraction"] >= c.stack_side_min_contact_fraction,
            "limited_overlap": max_overlap <= c.stack_side_max_overlap_fraction,
        }
        top_checks = {
            "smaller_foreground": c.stack_top_min_area_ratio <= ratio <= c.stack_top_max_area_ratio,
            "actual_mask_overlap": coverage >= c.stack_top_min_mask_coverage,
            "outer_boundary_contact": boundary["contact_fraction"] >= c.stack_top_min_outer_contact_fraction,
            "vertical_offset": c.stack_top_min_vertical_offset <= abs(dy) <= c.stack_top_max_vertical_offset,
            "horizontal_alignment": abs(dx) <= c.stack_top_max_horizontal_offset,
            "compact_objects": max(a.aspect_ratio, b.aspect_ratio) <= c.stack_top_max_aspect_ratio,
            "support_solidity": b.solidity >= c.stack_top_min_solidity,
        }
        side_ok, top_ok = all(side_checks.values()), all(top_checks.values())
        mode = "opposing_edges" if side_ok else "offset_overlap" if top_ok else None
        rank = (profile["contact_fraction"] if side_ok else coverage * boundary["contact_fraction"] * b.solidity
                if top_ok else 0.0)
        return {"source": a.id, "target": b.id, "mode": mode, "rank_score": rank,
                "source_offset_x_in_target_width": dx, "source_offset_y_in_target_height": dy,
                "opposing_edges_checks": side_checks, "offset_overlap_checks": top_checks,
                "selection": "eligible" if mode else "rejected"}

    def infer_relations(self, objects: list[MaskGeometry], pairs: list[dict[str, Any]],
                        scale: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        c = self.config
        by_id = {obj.id: obj for obj in objects}
        relations: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        directions: dict[tuple[str, str], list[tuple[float, str, int]]] = {}
        stacks: dict[str, list[tuple[dict[str, Any], int]]] = {}

        def add(source: str, target: str, relation: str, index: int, evidence: str,
                *, hypothesis: bool = False) -> None:
            key = (source, target, relation)
            if key in seen:
                return
            seen.add(key)
            input_review = bool(by_id[source].record.get("needs_review") or by_id[target].record.get("needs_review"))
            edge = {"source": source, "target": target, "type": relation, "measurements_index": index,
                    "evidence": evidence, "scope": "physical_hypothesis" if hypothesis else "image_plane",
                    "needs_review": hypothesis or input_review, "input_needs_review": input_review}
            if hypothesis:
                edge["review_reasons"] = ["single_view_depth_and_support_unobserved"]
            (candidates if hypothesis else relations).append(edge)

        def inverse(source: str, target: str, relation: str, index: int, evidence: str,
                    *, hypothesis: bool = False) -> None:
            add(source, target, relation, index, evidence, hypothesis=hypothesis)
            add(target, source, INVERSES[relation], index, evidence, hypothesis=hypothesis)

        for index, m in enumerate(pairs):
            a, b = by_id[m["a"]], by_id[m["b"]]
            a_inside = self._inside_method(m["fraction_a_in_b"], m["fraction_a_in_enclosed_b"],
                                           m["enclosed_area_ratio_b_to_a"])
            b_inside = self._inside_method(m["fraction_b_in_a"], m["fraction_b_in_enclosed_a"],
                                           1 / m["enclosed_area_ratio_b_to_a"])
            nested = a_inside is not None or b_inside is not None
            m["containment_methods"] = {"a_inside_b": a_inside, "b_inside_a": b_inside}
            if a_inside:
                inverse(a.id, b.id, "inside", index, a_inside)
            if b_inside:
                inverse(b.id, a.id, "inside", index, b_inside)
            max_fraction = max(m["fraction_a_in_b"], m["fraction_b_in_a"])
            if m["intersection_pixels"] >= c.overlap_min_pixels * scale**2 and max_fraction >= c.overlap_min_fraction:
                add(a.id, b.id, "overlaps", index, "actual_mask_intersection")
            contact = max(m["outer_boundary_a_to_b"]["contact_fraction"],
                          m["outer_boundary_b_to_a"]["contact_fraction"])
            if not nested and max_fraction <= c.touch_max_overlap_fraction and contact >= c.touch_min_boundary_fraction:
                add(a.id, b.id, "touching", index, "external_contour_distance")

            direction_eligible = not nested and max_fraction < c.direction_nested_fraction
            horizontal = (direction_eligible and abs(m["centroid_dx"]) > m["horizontal_dead_zone"]
                          and m["y_projection_overlap"] >= c.direction_projection_overlap
                          and abs(m["normalized_dx"]) <= c.direction_max_normalized_distance)
            vertical = (direction_eligible and abs(m["centroid_dy"]) > m["vertical_dead_zone"]
                        and m["x_projection_overlap"] >= c.direction_projection_overlap
                        and abs(m["normalized_dy"]) <= c.direction_max_normalized_distance)
            m["direction_eligibility"] = {"horizontal": horizontal, "vertical": vertical}
            if horizontal:
                left, right = (a.id, b.id) if m["centroid_dx"] > 0 else (b.id, a.id)
                for source, target, relation in ((left, right, "left_of"), (right, left, "right_of")):
                    directions.setdefault((source, relation), []).append((abs(m["centroid_dx"]), target, index))
            if vertical:
                upper, lower = (a.id, b.id) if m["centroid_dy"] > 0 else (b.id, a.id)
                for source, target, relation in ((upper, lower, "above"), (lower, upper, "below")):
                    directions.setdefault((source, relation), []).append((abs(m["centroid_dy"]), target, index))

            m["stacking"] = [self._stack_evidence(a, b, m, True, nested),
                              self._stack_evidence(b, a, m, False, nested)]
            for proposal in m["stacking"]:
                if proposal["mode"]:
                    stacks.setdefault(proposal["source"], []).append((proposal, index))

        # Union of each object's nearest aligned neighbors; inverse edges remain consistent.
        for (source, relation), options in sorted(directions.items()):
            for _, target, index in sorted(options)[:c.direction_neighbors]:
                inverse(source, target, relation, index, "nearest_aligned_centroid")

        selected: dict[str, tuple[dict[str, Any], int]] = {}
        for source, proposals in stacks.items():
            ranked = sorted(proposals, key=lambda p: (-p[0]["rank_score"], p[0]["target"]))
            best = ranked[0][0]["rank_score"]
            ambiguous = len(ranked) > 1 and best - ranked[1][0]["rank_score"] <= best * c.stack_ambiguity_fraction
            for position, (proposal, index) in enumerate(ranked):
                proposal["selection"] = "ambiguous" if ambiguous else "selected" if position == 0 else "weaker_candidate"
                if not ambiguous and position == 0:
                    selected[source] = (proposal, index)
        # A proposed support graph must also be acyclic, even across pattern types.
        cyclic: set[str] = set()
        for start in selected:
            path: list[str] = []
            current = start
            while current in selected and current not in path:
                path.append(current)
                current = selected[current][0]["target"]
            if current in path:
                cyclic.update(path[path.index(current):])
        for source, (proposal, index) in selected.items():
            if source in cyclic:
                proposal["selection"] = "cyclic_abstention"
            else:
                inverse(source, proposal["target"], "on", index, proposal["mode"], hypothesis=True)
        relations.sort(key=lambda edge: (edge["source"], edge["type"], edge["target"]))
        candidates.sort(key=lambda edge: (edge["source"], edge["type"], edge["target"]))
        return relations, candidates

    def analyze_scene(self, scene: dict[str, Any], debug_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        name = _component(scene["scene"], "scene name")
        if self.dataset_dir is not None:
            image_path = self.resolve(self.dataset_dir / name / "rgb.png")
        else:
            image_path = self.resolve(scene.get("image_path", f"dataset/{name}/rgb.png"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"{name}: cannot load image {image_path}")
        height, width = image.shape[:2]
        objects = [self.load_mask(record, (height, width)) for record in scene["objects"]]
        scale = max(height, width) / self.config.reference_image_side
        tolerance = max(self.config.touch_min_pixels, self.config.touch_tolerance_pixels * scale)
        boundaries = self.measure_boundaries(objects, (height, width), tolerance)
        pairs = [self.measure_pair(a, b, scale=scale, boundaries=boundaries)
                 for i, a in enumerate(objects) for b in objects[i+1:]]
        for index, pair in enumerate(pairs):
            pair["index"] = index
        relations, candidates = self.infer_relations(objects, pairs, scale)
        result = deepcopy(scene)
        result["image"] = {"width": width, "height": height}
        result["geometry_version"] = "1.0"
        result["relations"] = relations
        result["relation_candidates"] = candidates
        for record in result["objects"]:
            record["geometric_relations"] = {name: [] for name in PIXEL_RELATIONS}
            record["geometric_candidates"] = {"on": [], "under": []}
        records = {record["id"]: record for record in result["objects"]}
        for edge in relations:
            records[edge["source"]]["geometric_relations"][edge["type"]].append(edge["target"])
            if edge["type"] in ("overlaps", "touching"):
                records[edge["target"]]["geometric_relations"][edge["type"]].append(edge["source"])
        for edge in candidates:
            records[edge["source"]]["geometric_candidates"][edge["type"]].append(edge["target"])
        debug = {"scene": name, "image": result["image"], "pixel_scale": scale,
                 "effective_touch_tolerance": tolerance,
                 "objects": [{"id": obj.id, "mask_box": list(obj.bounds), "mask_area": obj.area,
                              "enclosed_area": obj.enclosed_area, "solidity": obj.solidity,
                              "aspect_ratio": obj.aspect_ratio, "outer_boundary_pixels": int(obj.boundary_x.size)}
                             for obj in objects], "pairs": pairs}
        if debug_dir is not None:
            self.visualize(image, objects, result, Path(debug_dir))
        return result, debug

    @staticmethod
    def save_results(data: Any, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    @staticmethod
    def visualize(image: np.ndarray, objects: list[MaskGeometry], result: dict[str, Any], directory: Path) -> None:
        """Separate structural/directional views with numbered arrows and full legends."""
        palette = [(80, 180, 255), (255, 160, 90), (120, 220, 100), (230, 120, 230), (80, 220, 220)]
        overlay = image.copy()
        for i, obj in enumerate(objects):
            color = palette[i % len(palette)]
            x1, y1, x2, y2 = obj.bounds
            region = overlay[y1:y2, x1:x2]
            selected = obj.mask.astype(bool)
            region[selected] = (region[selected] * 0.75 + np.array(color) * 0.25).astype(np.uint8)
            contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2, offset=(x1, y1))
        factor = min(1.0, 1400 / max(image.shape[:2]))
        size = (round(image.shape[1] * factor), round(image.shape[0] * factor))
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
        points: dict[str, tuple[int, int]] = {}
        for i, obj in enumerate(objects):
            color = palette[i % len(palette)]
            safe = tuple(round(v * factor) for v in obj.record["safe_pixel"])
            points[obj.id] = tuple(round(v * factor) for v in obj.record["centroid"])
            cv2.drawMarker(overlay, safe, (0, 255, 255), cv2.MARKER_STAR, 16, 2)
            cv2.circle(overlay, points[obj.id], 4, color, -1)
            text_width = cv2.getTextSize(obj.id, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            tx = min(max(0, safe[0] + 9), max(0, size[0] - text_width - 4))
            ty = min(max(18, safe[1] - 12), size[1] - 4)
            cv2.rectangle(overlay, (tx, ty-15), (tx+text_width+3, ty+3), (20, 20, 20), -1)
            cv2.putText(overlay, obj.id, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        directory.mkdir(parents=True, exist_ok=True)
        for direction_view in (False, True):
            allowed = {"left_of", "above"} if direction_view else {"inside", "overlaps", "touching", "on"}
            edges = [edge for edge in result["relations"] + result["relation_candidates"] if edge["type"] in allowed]
            title = "Nearest aligned directions" if direction_view else "Mask geometry; on? = unverified support"
            panel_width = max(600, max((len(f"{e['source']} {e['type']} {e['target']}") for e in edges), default=0) * 9 + 70)
            canvas_height = max(size[1] + 50, 115 + 26 * len(edges))
            canvas = np.full((canvas_height, size[0] + panel_width, 3), 250, np.uint8)
            canvas[50:50+size[1], :size[0]] = overlay
            cv2.putText(canvas, f"{result['scene']} - {title}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Stars: safe pixels. Dots: centroids.", (size[0]+15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
            for i, edge in enumerate(edges, 1):
                color = (190, 30, 190) if edge["type"] == "on" else (30, 80, 20)
                source = (points[edge["source"]][0], points[edge["source"]][1] + 50)
                target = (points[edge["target"]][0], points[edge["target"]][1] + 50)
                cv2.arrowedLine(canvas, source, target, color, 1, cv2.LINE_AA, tipLength=0.04)
                mid = ((source[0]+target[0])//2, (source[1]+target[1])//2 + (i % 3)*13)
                cv2.putText(canvas, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(canvas, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                relation = "on?" if edge["type"] == "on" else edge["type"]
                text = f"{i}. {edge['source']} {relation} {edge['target']}"
                cv2.putText(canvas, text, (size[0]+15, 90 + (i-1)*26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            if not edges:
                cv2.putText(canvas, "No relations passed these checks.", (size[0]+15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
            suffix = "_directions" if direction_view else ""
            path = directory / f"{result['scene']}{suffix}.png"
            if not cv2.imwrite(str(path), canvas):
                raise OSError(f"Cannot write geometry visualization {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/grounded_pipeline/segmentation_results.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grounded_pipeline/geometry_results.json"))
    parser.add_argument("--debug-output", type=Path, default=Path("outputs/grounded_pipeline/geometry_debug.json"))
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/grounded_pipeline/geometry_debug"))
    parser.add_argument("--base-dir", type=Path, help="Root of input mask/image paths; defaults to input JSON's directory")
    parser.add_argument("--dataset", type=Path, help="Override image paths with <dataset>/<scene>/rgb.png")
    parser.add_argument("--config", type=Path, help="JSON object overriding GeometryConfig fields")
    parser.add_argument("--scene", action="append", help="Process only this scene; may be repeated")
    parser.add_argument("--no-visualizations", action="store_true")
    args = parser.parse_args(argv)
    overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(overrides, dict):
        raise ValueError("Config must be a JSON object")
    config = GeometryConfig(**overrides)
    reasoner = GeometryReasoner(config, base_dir=args.base_dir or args.input.resolve().parent, dataset_dir=args.dataset)
    scenes = reasoner.load_detections(args.input)
    if args.scene:
        unknown = set(args.scene) - {scene["scene"] for scene in scenes}
        if unknown:
            raise ValueError(f"Unknown scenes: {sorted(unknown)}")
        scenes = [scene for scene in scenes if scene["scene"] in args.scene]
    started = time.perf_counter()
    results, debug_scenes = [], []
    for scene in scenes:
        result, debug = reasoner.analyze_scene(scene, None if args.no_visualizations else args.debug_dir)
        results.append(result)
        debug_scenes.append(debug)
        print(f"{scene['scene']}: {len(result['objects'])} objects, {len(result['relations'])} image relations, "
              f"{sum(e['type'] == 'on' for e in result['relation_candidates'])} support hypotheses")
    reasoner.save_results(results, args.output)
    reasoner.save_results({"config": asdict(config), "scenes": debug_scenes}, args.debug_output)
    print(f"Saved {args.output} and {args.debug_output} in {time.perf_counter()-started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
