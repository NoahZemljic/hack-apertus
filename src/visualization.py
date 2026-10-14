"""Draw the SAM-derived scene graph and its safe picking pixels."""
from pathlib import Path

import cv2
import numpy as np


def write_scene_overlay(rgb: np.ndarray, graph: dict, path: Path) -> None:
    """Save the original-resolution image with a box, ID and target for each object."""
    # OpenCV expects BGR colors and draws on a separate copy of the source image
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale = max(canvas.shape[:2]) / 1600
    thickness = max(1, round(2 * scale))
    font_scale = max(0.6, 0.6 * scale)
    marker_size = max(10, round(18 * scale))
    colors = [(0, 220, 255), (255, 180, 0), (80, 230, 80), (220, 80, 220)]

    for index, obj in enumerate(graph["objects"]):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = obj["bbox_2d"]
        target_x, target_y = obj["pick_point_2d"]

        # Planner boxes use inclusive endpoints in the original image.
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.drawMarker(
            canvas, (target_x, target_y), color,
            cv2.MARKER_CROSS, marker_size, thickness,
        )

        label = obj["id"]
        # Outline the label in black so it remains readable on light backgrounds
        label_position = (x1, max(round(20 * font_scale), y1 - 8))
        cv2.putText(
            canvas, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, thickness, cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write scene overlay to {path}")
