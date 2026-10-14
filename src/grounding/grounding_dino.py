import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

if __package__:
    from .postprocess_grounding import process_scene
else:
    from postprocess_grounding import process_scene


MODEL_ID = "IDEA-Research/grounding-dino-base"


def normalize_queries(queries: list[str]) -> list[str]:
    """Normalize spelling and known object plurals, keeping first-seen order."""
    # Use explicit aliases rather than stripping 's' from arbitrary nouns.
    aliases = {
        "gray": "grey",
        "trays": "tray",
        "cubes": "cube",
        "gears": "gear",
        "parts": "part",
        "mugs": "mug",
        "plates": "plate",
        "saucers": "saucer",
        "mandarins": "mandarin",
        "bananas": "banana",
        "fruits": "fruit",
    }
    normalized = []
    seen = set()
    for query in queries:
        words = query.lower().strip(" .\t\r\n").split()
        canonical = " ".join(aliases.get(word, word) for word in words)
        if canonical and canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def assign_detection_ids(detections: list[dict]) -> list[dict]:
    """Return detections with unique label-based IDs in detection order.

    Counters reset for each scene. These raw IDs also trace boxes through cleanup.
    Labels are preserved; combined DINO phrases are not resolved here.
    """
    counts = Counter()
    identified = []
    for detection in detections:
        stem = re.sub(r"[^a-z0-9]+", "_", detection["label"].lower()).strip("_")
        if not stem or not stem[0].isalpha():
            stem = f"object_{stem}" if stem else "object"
        counts[stem] += 1
        identified.append({**detection, "id": f"{stem}_{counts[stem]}"})
    return identified


class GroundingDinoDetector:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

    @staticmethod
    def load_queries(
        path: Path,
        query_source: str = "qwen_objects",
    ) -> list[dict]:
        """
        Load scene queries from scene_discovery.json.

        select from:
            "qwen_objects"
            "grounding_queries"
        """
        data = json.loads(path.read_text())

        scenes = []

        for item in data:
            scenes.append(
                {
                    "scene": item["scene"],
                    "queries": normalize_queries(item[query_source]),
                }
            )

        return scenes

    def detect(
        self,
        image_path: Path,
        queries: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """Ground each verified category independently using the same loaded model.

        A joint caption can produce phrases containing multiple categories and
        suppress separate instances. Isolated captions keep the detected object
        category unambiguous; backend phrases remain available for diagnosis.
        """

        queries = normalize_queries(queries)
        if not queries:
            return []

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        import torch

        detections = []
        for query in queries:
            inputs = self.processor(
                images=image,
                text=f"{query}.",
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            for box, score, text_label in zip(
                results["boxes"],
                results["scores"],
                results["text_labels"],
            ):
                detections.append({
                    "label": query,
                    "query": query,
                    "text_label": text_label,
                    "box": [round(x, 2) for x in box.tolist()],
                    "score": round(score.item(), 4),
                })

        return assign_detection_ids(detections)

    @staticmethod
    def plot(
        image_path: Path,
        detections: list[dict],
        output_path: Path,
        title: str = "Raw DINO detections",
    ):
        """Draw detected boxes for debugging."""
        save_box_plot(image_path, [(title, detections)], output_path, detections)

    @staticmethod
    def plot_comparison(
        image_path: Path,
        raw_detections: list[dict],
        cleaned_detections: list[dict],
        output_path: Path,
    ) -> None:
        save_box_plot(
            image_path,
            [("Raw DINO", raw_detections), ("After cleanup (? = needs review)", cleaned_detections)],
            output_path,
            raw_detections,
        )

    @staticmethod
    def save_results(
        results: list[dict],
        output_path: Path,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


def save_box_plot(
    image_path: Path,
    panels: list[tuple[str, list[dict]]],
    output_path: Path,
    raw_detections: list[dict],
) -> None:
    """Save plots on a headless canvas, keeping source-box colors across panels."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    figure = Figure(figsize=(12 * len(panels), 10))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    palette = colormaps["tab20"]
    color_by_id = {d["id"]: palette(i % 20) for i, d in enumerate(raw_detections)}
    for ax, (title, detections) in zip(axes, panels):
        ax.imshow(image)
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["box"]
            source_id = (detection.get("source_ids") or [detection["id"]])[0]
            color = color_by_id.get(source_id, palette(index % 20))
            review = detection.get("needs_review", False)
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False,
                linewidth=2, edgecolor=color, linestyle="--" if review else "-",
            ))
            ax.text(
                max(0, x1), max(0, y1),
                f'{detection["id"]} {detection["score"]:.2f}' + (" ?" if review else ""),
                fontsize=9, color="white",
                bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none"},
            )
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.set_title(f"{title}: {len(detections)} boxes")
        ax.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description="Run DINO, conservative cleanup, and comparison plots")
    parser.add_argument("--queries", type=Path, default=Path("outputs/grounded_pipeline/scene_discovery.json"))
    parser.add_argument("--query-source", choices=["qwen_objects", "grounding_queries"], default="qwen_objects")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grounded_pipeline/grounding_detections.json"))
    parser.add_argument("--clean-output", type=Path, default=Path("outputs/grounded_pipeline/grounding_detections_clean.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/grounded_pipeline/grounding_cleanup_report.json"))
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/grounded_pipeline/grounding_debug"))
    parser.add_argument("--input-detections", type=Path, help="Clean and plot saved DINO results without running the model")
    parser.add_argument("--scene", help="Process only this scene, e.g. scene_01")
    parser.add_argument("--duplicate-iou", type=float, default=0.90)
    parser.add_argument("--max-object-area-ratio", type=float, default=0.35)
    args = parser.parse_args()
    if args.input_detections:
        scenes = json.loads(args.input_detections.read_text())
        # The saved-output path needs plotting/cleanup only, no model downloads.
        detector = GroundingDinoDetector
    else:
        detector = GroundingDinoDetector()
        scenes = detector.load_queries(args.queries, query_source=args.query_source)
    if args.scene:
        scenes = [item for item in scenes if item["scene"] == args.scene]
        if not scenes:
            parser.error(f"Scene {args.scene!r} is absent from the input")
    all_results = []
    clean_results = []
    reports = []

    for scene_item in scenes:
        scene = scene_item["scene"]
        queries = scene_item["queries"]

        image_path = args.dataset / scene / "rgb.png"

        print(f"Processing {scene}")
        print(f"Queries: {queries}")

        detections = (
            scene_item["detections"] if args.input_detections
            else detector.detect(image_path, queries)
        )
        raw_result = {"scene": scene, "queries": queries, "detections": detections}
        cleaned, report = process_scene(
            raw_result, image_path,
            duplicate_iou=args.duplicate_iou,
            max_object_area_ratio=args.max_object_area_ratio,
        )
        clean_results.append(cleaned)
        reports.append(report)
        print(f'Cleanup: {report["raw_count"]} -> {report["clean_count"]} boxes; '
              f'{len(report["review_ids"])} need review')

        print(
            "Found:",
            [
                f'{d["id"]} ({d["score"]:.2f})'
                for d in detections
            ],
        )

        all_results.append(raw_result)

        detector.plot(
            image_path,
            detections,
            args.debug_dir / f"{scene}_boxes.png",
        )
        detector.plot(
            image_path, cleaned["detections"],
            args.debug_dir / f"{scene}_boxes_clean.png",
            title="After cleanup (? = needs review)",
        )
        detector.plot_comparison(
            image_path, detections, cleaned["detections"],
            args.debug_dir / f"{scene}_comparison.png",
        )

    # Replaying saved detections leaves the source JSON intact.
    if not args.input_detections:
        detector.save_results(all_results, args.output)
        print(f"\nSaved raw detections to {args.output}")
    detector.save_results(clean_results, args.clean_output)
    detector.save_results(reports, args.report)
    print(f"Saved cleaned detections to {args.clean_output}")
    print(f"Saved cleanup report to {args.report}")
    print(f"Saved debug images to {args.debug_dir}")


if __name__ == "__main__":
    main()
