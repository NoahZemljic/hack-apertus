"""Measure detected objects and derive their image-space relations.

Qwen supplies labels, bounding boxes and target pixels. This module uses those
detections to estimate physical width and compute relations in Python.
"""

from collections import Counter
from itertools import combinations

import cv2
import numpy as np

from starter.scene_io import TABLE_Z_MM, pixel_to_robot

GRIPPER_WIDTH_MM = 85.0
MAX_SEGMENTATION_SIZE = 640


def object_contour(rgb: np.ndarray, bbox: list[int], target: list[int]) -> np.ndarray:
    """Return the segmented contour containing the target."""
    image_height, image_width = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    target_x, target_y = target

    # Check the box and target point
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("Cannot segment a box outside the image or with nonpositive area")
    if not (x1 <= target_x < x2 and y1 <= target_y < y2):
        raise ValueError("Cannot segment: target point is outside its box")

    # Add a small border around the detection box for GrabCut
    padding = max(4, round(max(x2 - x1, y2 - y1) * 0.05))
    crop_left = max(0, x1 - padding)
    crop_top = max(0, y1 - padding)
    crop_right = min(image_width, x2 + padding)
    crop_bottom = min(image_height, y2 + padding)
    crop = rgb[crop_top:crop_bottom, crop_left:crop_right]

    # Limit segmentation cost and track the scale back to source pixels
    scale = min(1.0, MAX_SEGMENTATION_SIZE / max(crop.shape[:2]))
    resized_width = max(1, round(crop.shape[1] * scale))
    resized_height = max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (resized_width, resized_height))
    scale_x = resized.shape[1] / crop.shape[1]
    scale_y = resized.shape[0] / crop.shape[0]

    # Mark the box as probable foreground and the target point as definite foreground
    box_left = int((x1 - crop_left) * scale_x)
    box_top = int((y1 - crop_top) * scale_y)
    box_right = int(np.ceil((x2 - crop_left) * scale_x))
    box_bottom = int(np.ceil((y2 - crop_top) * scale_y))
    target_point = (
        min(resized.shape[1] - 1, int((target_x - crop_left) * scale_x)),
        min(resized.shape[0] - 1, int((target_y - crop_top) * scale_y)),
    )

    mask = np.full(resized.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[box_top:box_bottom, box_left:box_right] = cv2.GC_PR_FGD
    mask[target_point[1], target_point[0]] = cv2.GC_FGD
    if not np.any(mask == cv2.GC_BGD):
        raise ValueError("Cannot segment without background outside the detection box")

    # Fix GrabCut's random seed so repeated runs produce consistent measurements
    cv2.setRNGSeed(0)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        resized,
        mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )

    # Keep the connected foreground region that contains the target point
    foreground = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    _, component_labels = cv2.connectedComponents(foreground)
    target_component = component_labels[target_point[1], target_point[0]]
    component_mask = (component_labels == target_component).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea).astype(np.float32)
    if cv2.contourArea(contour) < 4:
        raise ValueError("The target pixel did not identify a measurable foreground region")

    # Convert the contour from resized-crop coordinates to original-image coordinates
    contour[:, 0, 0] = contour[:, 0, 0] / scale_x + crop_left
    contour[:, 0, 1] = contour[:, 0, 1] / scale_y + crop_top
    return contour


def project_to_object_plane(scene, points: np.ndarray, target: list[int]) -> np.ndarray:
    """Project contour pixels onto a horizontal plane in robot-base coordinates.

    The target's measured height is used when depth is available. Otherwise the
    calibrated table plane is used, so widths of raised objects are approximate.
    """
    if scene.base_T_camera is None:
        raise ValueError("Physical width requires camera-to-robot calibration")

    # Choose the measurement plane from depth when available, or from calibration otherwise
    plane_z = TABLE_Z_MM
    if scene.xyz is not None or scene.depth_mm is not None:
        plane_z = float(pixel_to_robot(scene, *target)[2])

    # Convert each pixel into a camera ray, then express the ray in robot coordinates
    camera_to_base = scene.base_T_camera
    pixels = np.column_stack((points.reshape(-1, 2), np.ones(len(points))))
    rays = pixels @ np.linalg.inv(scene.K).T @ camera_to_base[:3, :3].T

    # Intersect each ray with the selected horizontal plane in front of the camera
    if np.any(np.abs(rays[:, 2]) < 1e-9):
        raise ValueError("Camera rays are parallel to the measurement plane")
    distances = (plane_z - camera_to_base[2, 3]) / rays[:, 2]
    if np.any(distances <= 0):
        raise ValueError("Measurement plane is behind the camera")
    positions = camera_to_base[:3, 3] + rays * distances[:, None]
    if not np.isfinite(positions).all():
        raise ValueError("Projection produced non-finite coordinates")

    return positions[:, :2].astype(np.float32)


def measure_width(scene, contour: np.ndarray, target: list[int]) -> float:
    """Measure the shorter side of the contour's minimum-area rectangle in mm."""
    physical_contour = project_to_object_plane(scene, contour, target)
    _, (side_a, side_b), _ = cv2.minAreaRect(physical_contour)
    width_mm = float(min(side_a, side_b))

    if not np.isfinite(width_mm) or width_mm <= 0:
        raise ValueError("Minimum-area rectangle has no measurable width")

    return width_mm


def compute_objects(scene, detections: dict) -> list[dict]:
    """Assign stable per-label IDs and add width and graspability when measurable.

    A failed measurement leaves that object's detection available to the scene
    graph, but omits its physical width and graspability.
    """
    label_counts = Counter()
    objects = []

    for detection in detections["objects"]:
        label = detection["label"]
        label_counts[label] += 1
        obj = {**detection, "id": f"{label}_{label_counts[label]}"}

        try:
            contour = object_contour(scene.rgb, obj["bbox_2d"], obj["target_pixel"])
            width_mm = measure_width(scene, contour, obj["target_pixel"])

            obj["width_mm"] = width_mm
            obj["graspable"] = width_mm <= GRIPPER_WIDTH_MM
            obj["rectangle_2d"] = cv2.boxPoints(cv2.minAreaRect(contour)).tolist()
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            obj["measurement_error"] = str(error)

        objects.append(obj)

    return objects


def append_relation(
    relations: list[dict],
    subject: dict,
    relation: str,
    target: dict,
    confidence: float,
) -> None:
    """Append one computed relation with a four-decimal confidence score."""
    relations.append({
        "subject": subject["id"],
        "relation": relation,
        "object": target["id"],
        "confidence": round(float(confidence), 4),
    })


def compute_relations(objects: list[dict]) -> list[dict]:
    """Derive left/right, above/below and tray containment from detections.

    Direction confidence is separation divided by separation plus the pair's
    mean box extent. Inside confidence is the target's nearest tray-edge margin
    divided by half the tray's shorter side. These are geometric scores, not
    calibrated probabilities.
    """
    relations = []

    # Compare each pair of target pixels along the horizontal and vertical axes
    directions = ((0, "left", "right"), (1, "above", "below"))
    for first, second in combinations(objects, 2):
        for axis, forward, reverse in directions:
            separation = second["target_pixel"][axis] - first["target_pixel"][axis]
            if separation == 0:
                continue

            first_extent = abs(first["bbox_2d"][axis + 2] - first["bbox_2d"][axis])
            second_extent = abs(second["bbox_2d"][axis + 2] - second["bbox_2d"][axis])
            mean_extent = (first_extent + second_extent) / 2
            confidence = abs(separation) / (abs(separation) + mean_extent)
            relation, inverse = (forward, reverse) if separation > 0 else (reverse, forward)

            # Store both directions so every object has relations to its neighbors
            append_relation(relations, first, relation, second, confidence)
            append_relation(relations, second, inverse, first, confidence)

    # A tray is identified by a label ending in "_tray"; its box defines containment
    trays = [obj for obj in objects if obj["label"].split("_")[-1] == "tray"]
    for obj in objects:
        target_x, target_y = obj["target_pixel"]
        for tray in trays:
            if obj["id"] == tray["id"]:
                continue

            x1, y1, x2, y2 = tray["bbox_2d"]
            if not (x1 <= target_x < x2 and y1 <= target_y < y2):
                continue

            # Use the nearest box edge as the containment margin
            margin = min(target_x - x1, x2 - target_x, target_y - y1, y2 - target_y)
            half_shorter_side = min(x2 - x1, y2 - y1) / 2
            confidence = min(1.0, margin / half_shorter_side)
            append_relation(relations, obj, "inside", tray, confidence)

    return relations
