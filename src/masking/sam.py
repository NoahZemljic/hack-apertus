"""Segment cleaned DINO boxes with SAM 2 and calculate mask geometry.

Run from the repository root: python -m src.masking.sam
"""

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


MODEL_ID = "facebook/sam2.1-hiera-small"
DEFAULT_DETECTIONS = Path("outputs/grounded_pipeline/grounding_detections_clean.json")
Mask = NDArray[np.bool_]


def validate_component(value: str, name: str) -> None:
    """Require a single path component for scene directories and mask files."""
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be a nonempty filename component: {value!r}")


def clip_box(box: list[float], image_size: tuple[int, int]) -> list[float]:
    """Validate an xyxy prompt; only the prompt is clipped, never its metadata."""
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError(f"Expected four finite xyxy coordinates, got {box}")
    width, height = image_size
    clipped = [
        max(0.0, min(float(value), limit))
        for value, limit in zip(box, [width, height, width, height])
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Box has no area inside the image: {box}")
    return clipped


class SamSegmenter:
    """One best nonempty SAM 2 mask per cleaned physical detection.

    Model loading is deferred until segmentation. Geometry, saving, validation,
    and visualization can be used without downloading a checkpoint.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        min_box_fraction: float = 0.02,
        max_outside_fraction: float = 0.10,
        outside_box_margin: float = 0.05,
    ) -> None:
        for name, value in [
            ("min_box_fraction", min_box_fraction),
            ("max_outside_fraction", max_outside_fraction),
            ("outside_box_margin", outside_box_margin),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.model_id = model_id
        self.device = device
        self.min_box_fraction = min_box_fraction
        self.max_outside_fraction = max_outside_fraction
        self.outside_box_margin = outside_box_margin
        self.processor: Any = None
        self.model: Any = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()

    @staticmethod
    def load_detections(path: Path = DEFAULT_DETECTIONS) -> list[dict[str, Any]]:
        """Load cleaned DINO scenes, requiring their metadata and unique IDs."""
        scenes = json.loads(path.read_text())
        if not isinstance(scenes, list):
            raise ValueError("Expected a list of cleaned DINO scene records")
        seen_scenes = set()
        required = {"id", "label", "aliases", "score", "box"}
        for item in scenes:
            scene = item["scene"]
            validate_component(scene, "Scene")
            if scene in seen_scenes:
                raise ValueError(f"Duplicate scene: {scene}")
            seen_scenes.add(scene)
            if not isinstance(item.get("detections"), list):
                raise ValueError(f"{scene}: expected a detections list")
            seen_ids = set()
            for detection in item["detections"]:
                missing = required - detection.keys()
                if missing:
                    raise ValueError(f"{scene}: missing {sorted(missing)}; use cleaned DINO detections")
                validate_component(detection["id"], "Object ID")
                if detection["id"] in seen_ids:
                    raise ValueError(f"{scene}: duplicate object ID {detection['id']!r}")
                seen_ids.add(detection["id"])
                if not isinstance(detection["label"], str) or not detection["label"].strip():
                    raise ValueError(f"{scene}: detection label must be a nonempty string")
                if not isinstance(detection["aliases"], list) or not all(
                    isinstance(alias, str) for alias in detection["aliases"]
                ):
                    raise ValueError(f"{scene}: aliases must be a list of strings")
                if not math.isfinite(detection["score"]):
                    raise ValueError(f"{scene}: score must be finite")
        return scenes

    @staticmethod
    def calculate_geometry(mask: Mask) -> dict[str, Any]:
        """Calculate geometry in full-image pixels, with xyxy mask bounds.

        The centroid is the arithmetic mean of foreground coordinates and can
        lie in a hole. The safe pixel maximizes the exact Euclidean distance to
        background. Padding treats the image edge as a mask boundary too.
        """
        import cv2

        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("Mask must be a nonempty two-dimensional array")
        ys, xs = np.nonzero(mask)
        distances = cv2.distanceTransform(
            np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
        )[1:-1, 1:-1]
        safe_y, safe_x = np.unravel_index(np.argmax(distances), mask.shape)
        return {
            "mask_area": int(xs.size),
            "mask_box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid": [float(xs.mean()), float(ys.mean())],
            "safe_pixel": [int(safe_x), int(safe_y)],
            "safe_pixel_clearance": float(distances[safe_y, safe_x]),
        }

    def validate_mask(
        self, mask: Mask, box: list[float], geometry: dict[str, Any],
    ) -> dict[str, Any]:
        """Reject invalid geometry and report small masks or significant spill.

        Masks are never cropped to their prompts: spill must remain visible.
        """
        if mask.ndim != 2 or not mask.any():
            raise ValueError("SAM mask is empty or not two-dimensional")
        height, width = mask.shape
        safe_x, safe_y = geometry["safe_pixel"]
        if not (0 <= safe_x < width and 0 <= safe_y < height and mask[safe_y, safe_x]):
            raise ValueError("Safe pixel is outside the mask")
        cx, cy = geometry["centroid"]
        if not (0 <= cx < width and 0 <= cy < height):
            raise ValueError("Mask centroid is outside the image")
        x1, y1, x2, y2 = clip_box(box, (width, height))
        area = int(mask.sum())

        def count_inside(bounds: list[float]) -> int:
            left, top, right, bottom = bounds
            left, top = max(0, math.floor(left)), max(0, math.floor(top))
            right, bottom = min(width, math.ceil(right)), min(height, math.ceil(bottom))
            return int(mask[top:bottom, left:right].sum())

        inside = count_inside([x1, y1, x2, y2])
        box_fraction = inside / ((x2 - x1) * (y2 - y1))
        margin_x = (x2 - x1) * self.outside_box_margin
        margin_y = (y2 - y1) * self.outside_box_margin
        near_box = count_inside([x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y])
        outside_fraction = (area - inside) / area
        far_outside_fraction = (area - near_box) / area
        messages = []
        if box_fraction < self.min_box_fraction:
            messages.append(f"Mask fills only {box_fraction:.1%} of its DINO box")
        if far_outside_fraction > self.max_outside_fraction:
            messages.append(f"{far_outside_fraction:.1%} of mask pixels lie far outside its DINO box")
        if list(box) != [x1, y1, x2, y2]:
            messages.append("DINO prompt was clipped to image bounds; original box metadata is preserved")
        return {
            "box_mask_fraction": box_fraction,
            "outside_box_fraction": outside_fraction,
            "far_outside_box_fraction": far_outside_fraction,
            "validation_warnings": messages,
        }

    def segment_object(
        self,
        detection: dict[str, Any],
        candidate_masks: NDArray[np.bool_],
        candidate_scores: NDArray[np.floating],
    ) -> dict[str, Any]:
        """Select the highest-scoring nonempty candidate and retain all metadata."""
        masks = np.asarray(candidate_masks, dtype=bool)
        scores = np.asarray(candidate_scores, dtype=float)
        if masks.ndim != 3 or scores.shape != (len(masks),):
            raise ValueError(f"{detection['id']}: inconsistent SAM candidate shapes")
        valid = masks.any(axis=(1, 2)) & np.isfinite(scores)
        if not valid.any():
            raise ValueError(f"{detection['id']}: SAM produced no nonempty mask with a finite score")
        index = int(np.argmax(np.where(valid, scores, -np.inf)))
        mask = masks[index].copy()
        geometry = self.calculate_geometry(mask)
        validation = self.validate_mask(mask, detection["box"], geometry)
        for message in validation["validation_warnings"]:
            warnings.warn(f"{detection['id']}: {message}", RuntimeWarning, stacklevel=2)
        return {
            **detection,
            "dino_score": detection["score"],
            "sam_score": float(scores[index]),
            **geometry,
            **validation,
            "mask": mask,
        }

    def segment_scene(self, image_path: Path, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch every cleaned box in a scene through one image-encoder pass."""
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        boxes = [clip_box(detection["box"], image.size) for detection in detections]
        if not detections:
            return []
        self._load_model()
        import torch

        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=True)
        predictions = outputs.pred_masks.detach().cpu()
        scores = outputs.iou_scores[0].detach().cpu().numpy()
        if predictions.shape[1] != len(detections) or scores.shape[0] != len(detections):
            raise ValueError("SAM returned a different number of prompts than DINO detections")
        objects = []
        for index, detection in enumerate(detections):
            # Restore candidates one object at a time to limit full-size memory.
            masks = self.processor.post_process_masks(
                predictions[:, index:index + 1], inputs["original_sizes"].cpu(),
            )[0][0].numpy().astype(bool)
            if masks.shape[1:] != (image.height, image.width):
                raise ValueError(f"{detection['id']}: SAM mask is not at the original image resolution")
            objects.append(self.segment_object(detection, masks, scores[index]))
        return objects

    @staticmethod
    def save_mask(mask: Mask, output_path: Path) -> Path:
        """Write a compressed, full-resolution binary PNG instead of JSON pixels."""
        if mask.ndim != 2 or not mask.any():
            raise ValueError("Cannot save an empty or non-two-dimensional mask")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_path, compress_level=6)
        return output_path

    @staticmethod
    def visualize(image_path: Path, objects: list[dict[str, Any]], output_path: Path) -> None:
        """Show original RGB beside transparent masks, IDs, prompts, and safe pixels."""
        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        figure = Figure(figsize=(20, 9))
        FigureCanvasAgg(figure)
        original, annotated = figure.subplots(1, 2)
        for ax in [original, annotated]:
            ax.imshow(image)
            ax.axis("off")
        original.set_title("Original image")
        annotated.set_title(f"SAM 2: {len(objects)} masks (stars = safe pixels)")
        colors = {obj["id"]: colormaps["tab20"](i % 20) for i, obj in enumerate(objects)}
        overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        for obj in sorted(objects, key=lambda obj: obj["mask_area"], reverse=True):
            overlay[obj["mask"]] = [*np.round(np.array(colors[obj["id"]][:3]) * 255).astype(np.uint8), 100]
        annotated.imshow(overlay)
        for obj in objects:
            color = colors[obj["id"]]
            x1, y1, x2, y2 = obj["box"]
            annotated.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.2,
            ))
            sx, sy = obj["safe_pixel"]
            annotated.scatter(sx, sy, marker="*", s=100, c="yellow", edgecolors="black", linewidths=0.8)
            annotated.text(
                max(0, x1), max(0, y1), obj["id"], fontsize=9, color="white",
                bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none"},
            )
        annotated.set_xlim(0, image.width)
        annotated.set_ylim(image.height, 0)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        figure.clear()

    @staticmethod
    def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
        """Serialize scene metadata without in-memory mask matrices."""
        serializable = [
            {**scene, "objects": [{k: v for k, v in obj.items() if k != "mask"} for obj in scene["objects"]]}
            for scene in results
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grounded_pipeline/segmentation"))
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/grounded_pipeline/segmentation_debug"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grounded_pipeline/segmentation_results.json"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda:0; auto by default")
    parser.add_argument("--scene", help="Process only this scene, e.g. scene_01")
    parser.add_argument("--min-box-fraction", type=float, default=0.02)
    parser.add_argument("--max-outside-fraction", type=float, default=0.10)
    parser.add_argument("--outside-box-margin", type=float, default=0.05)
    args = parser.parse_args()

    scenes = SamSegmenter.load_detections(args.detections)
    if args.scene:
        scenes = [item for item in scenes if item["scene"] == args.scene]
        if not scenes:
            parser.error(f"Scene {args.scene!r} is absent from {args.detections}")
    segmenter = SamSegmenter(
        args.model_id, args.device, args.min_box_fraction,
        args.max_outside_fraction, args.outside_box_margin,
    )
    results = []
    for item in scenes:
        scene = item["scene"]
        image_path = args.dataset / scene / "rgb.png"
        print(f"Processing {scene}: {len(item['detections'])} cleaned boxes", flush=True)
        try:
            objects = segmenter.segment_scene(image_path, item["detections"])
        except ValueError as exc:
            raise ValueError(f"{scene}: {exc}") from exc
        for obj in objects:
            mask_path = segmenter.save_mask(obj["mask"], args.output_dir / scene / f"{obj['id']}.png")
            obj["mask_path"] = mask_path.resolve().as_posix()
        segmenter.visualize(image_path, objects, args.debug_dir / f"{scene}.png")
        results.append({
            **{key: value for key, value in item.items() if key != "detections"},
            "image_path": image_path.resolve().as_posix(),
            "objects": [{key: value for key, value in obj.items() if key != "mask"} for obj in objects],
        })
    segmenter.save_results(results, args.output)
    print(f"Saved segmentation metadata to {args.output}")
    print(f"Saved masks to {args.output_dir} and debug plots to {args.debug_dir}")


if __name__ == "__main__":
    main()
