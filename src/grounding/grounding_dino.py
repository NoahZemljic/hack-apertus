import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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

    Counters reset for each scene. Call after any future detection cleanup.
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
    def __init__(self, model_id: str = MODEL_ID):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    def load_queries(
        self,
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
        """Run Grounding DINO on one image."""

        queries = normalize_queries(queries)
        if not queries:
            return []

        image = Image.open(image_path).convert("RGB")

        # Grounding DINO works well with a dot-separated text prompt
        text = ". ".join(queries) + "."

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["text_labels"],
        ):
            detections.append(
                {
                    "label": label,
                    "box": [round(x, 2) for x in box.tolist()],
                    "score": round(score.item(), 4),
                }
            )

        return assign_detection_ids(detections)

    def plot(
        self,
        image_path: Path,
        detections: list[dict],
        output_path: Path,
    ):
        """Draw detected boxes for debugging."""

        image = Image.open(image_path).convert("RGB")

        plt.figure(figsize=(12, 10))
        plt.imshow(image)

        ax = plt.gca()

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            width = x2 - x1
            height = y2 - y1

            rectangle = plt.Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                linewidth=2,
            )

            ax.add_patch(rectangle)

            ax.text(
                x1,
                y1,
                f'{detection["id"]} {detection["score"]:.2f}',
                fontsize=9,
                bbox={"alpha": 0.7},
            )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def save_results(
        self,
        results: list[dict],
        output_path: Path,
    ):
        output_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    query_file = Path("scene_discovery.json")
    dataset = Path("dataset")

    output_file = Path("grounding_detections.json")
    debug_dir = Path("outputs/grounding")

    debug_dir.mkdir(parents=True, exist_ok=True)

    detector = GroundingDinoDetector()

    scenes = detector.load_queries(
        query_file,
        query_source="qwen_objects",
    )

    all_results = []

    for scene_item in scenes:
        scene = scene_item["scene"]
        queries = scene_item["queries"]

        image_path = dataset / scene / "rgb.png"

        print(f"Processing {scene}")
        print(f"Queries: {queries}")

        detections = detector.detect(
            image_path,
            queries,
        )

        print(
            "Found:",
            [
                f'{d["id"]} ({d["score"]:.2f})'
                for d in detections
            ],
        )

        all_results.append(
            {
                "scene": scene,
                "queries": queries,
                "detections": detections,
            }
        )

        detector.plot(
            image_path,
            detections,
            debug_dir / f"{scene}_boxes.png",
        )

    detector.save_results(
        all_results,
        output_file,
    )

    print(f"\nSaved detections to {output_file}")
    print(f"Saved debug images to {debug_dir}")


if __name__ == "__main__":
    main()
