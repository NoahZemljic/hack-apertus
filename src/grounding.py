"""Image-based material checks; no dataset annotations or scene-specific coordinates."""
import cv2
import numpy as np


def refine_material_point(bgr, obj: dict, objects: list[dict]) -> dict | None:
    """Keep or replace a pale industrial part's point using a high-contrast material mask.

    This narrow check abstains on low-contrast images. It is not a general object
    segmenter or a physical grasp validator. Dark holes stay excluded, and boxes
    of potential occluders are excluded conservatively.
    """
    if obj["label"] not in {"metal part", "bracket", "industrial component"}:
        return None
    color = (obj.get("color") or "").lower()
    if not any(name in color for name in ("white", "gray", "grey", "silver")):
        return None
    x1, y1, x2, y2 = obj["bbox_2d"]
    crop = bgr[y1:y2, x1:x2]
    if not crop.size:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    material = (gray > threshold) & (saturation < 80)
    dark = gray <= threshold
    if not material.any() or not dark.any():
        return None
    contrast = float(gray[material].mean() - gray[dark].mean())
    if contrast < 25:
        return None

    for other in objects:
        if (other is obj or other["kind"] == "container" or other["label"].endswith("mat")
                or other["id"] in obj["relations"]["on_top_of"]):
            continue
        a, b, c, d = other["bbox_2d"]
        left, top, right, bottom = max(x1, a), max(y1, b), min(x2, c), min(y2, d)
        if left < right and top < bottom:
            material[top-y1:bottom-y1, left-x1:right-x1] = False

    # A zero border makes distance-to-edge meaningful even for a tight crop.
    padded = np.pad(material.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    required_margin = max(3, min(crop.shape[:2]) * 0.04)
    if float(distance.max()) < required_margin:
        return None
    u, v = obj["pick_point_2d"]
    current_margin = float(distance[v-y1, u-x1]) if x1 <= u < x2 and y1 <= v < y2 else 0.0
    if current_margin >= required_margin:
        selected = [u, v]
        selected_margin = current_margin
    else:
        local_v, local_u = np.unravel_index(int(distance.argmax()), distance.shape)
        selected = [int(local_u) + x1, int(local_v) + y1]
        selected_margin = float(distance[local_v, local_u])
    return {"stage": "material_point", "object_id": obj["id"], "proposed": [u, v],
            "selected": selected, "margin_px": round(selected_margin, 2),
            "contrast": round(contrast, 2), "threshold": threshold}
