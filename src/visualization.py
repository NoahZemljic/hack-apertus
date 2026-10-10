"""Inspection overlays; measurement diagnostics never enter the planner's graph."""
from pathlib import Path

import cv2
import numpy as np


def write_scene_overlay(rgb, graph: dict, path: Path, geometry: list[dict] | None = None) -> None:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    measurements = {obj['id']: obj for obj in geometry or []}
    scale = max(canvas.shape[:2]) / 1600
    thickness = max(1, round(2 * scale))
    for obj in graph['objects']:
        color = (0, 180, 0) if obj.get('graspable') else (0, 160, 255)
        u, v = obj['target_pixel']
        details = measurements.get(obj['id'], {})
        label_position = (u, max(20, v - 8))
        if 'bbox_2d' in details:
            x1, y1, x2, y2 = details['bbox_2d']
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
            label_position = (x1, max(20, y1 - 8))
        if 'rectangle_2d' in details:
            points = np.rint(details['rectangle_2d']).astype(np.int32)
            cv2.polylines(canvas, [points], True, (255, 200, 0), thickness)
        cv2.drawMarker(canvas, (u, v), color, cv2.MARKER_CROSS, max(6, round(18 * scale)), thickness)
        measurement = f"{obj['width_mm']:.1f} mm" if 'width_mm' in obj else 'width unknown'
        cv2.putText(canvas, f"{obj['id']} / {measurement}", label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.6 * scale), color, thickness, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f'could not write scene overlay to {path}')
