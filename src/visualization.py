from pathlib import Path

import cv2

COLORS = {
    "movable": (0, 200, 0),
    "container": (255, 120, 0),
    "surface": (200, 0, 200),
    "human": (0, 0, 255),
}

def write_scene_overlay(rgb, graph: dict, path: Path) -> None:
    """Draw detected boxes, IDs, and pick points onto a scene image"""
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Scale labels and markers so overlays remain readable at full resolution
    scale = max(canvas.shape[:2]) / 1600
    thickness = max(2, round(2 * scale))
    font_scale = max(0.6, 0.7 * scale)

    for obj in graph["objects"]:
        color = COLORS[obj["kind"]]
        x1, y1, x2, y2 = obj["bbox_2d"]
        u, v = obj["pick_point_2d"]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.drawMarker(canvas, (u, v), color, cv2.MARKER_CROSS, round(18 * scale), thickness)
        cv2.putText(canvas, obj["id"], (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write scene overlay to {path}")
