import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import cv2

from .json_tools import parse_json_object
from .compute_geometry import stacking_candidates
from .schemas import load_perception_schema, load_schema
from .validation import validate_structure


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "qwen_system.txt"
STACKING_PROMPT_PATH = PROMPT_PATH.with_name("qwen_stacking.txt")

class PerceptionGenerationError(ValueError):
    """Keep the model's replies and errors when all detection attempts fail."""

    def __init__(self, message: str, replies: list[str], errors: list[str]) -> None:
        super().__init__(message)

        self.replies = replies
        self.errors = errors


class PerceptionModel(Protocol):
    """Define the detection method expected by the pipeline."""

    def detect(
        self,
        image_path: Path,
        width: int,
        height: int,
    ) -> tuple[dict, list[str]]:
        """Return detections in source-image pixels, plus the raw model replies."""
        ...

    def detect_stacking(self, image_path: Path, objects: list[dict]) -> list[dict]:
        """Return visually supported on relations between detected objects"""
        ...


class QwenPerception:
    """Ask Qwen for detections, check the JSON, and convert coordinates."""

    def __init__(self, vlm, num_validation_retries: int = 2) -> None:
        if num_validation_retries < 0:
            raise ValueError("validation_retries must be nonnegative")

        self.vlm = vlm
        self.validation_retries = num_validation_retries

    def detect(
        self,
        image_path: Path,
        width: int,
        height: int,
    ) -> tuple[dict, list[str]]:
        """Detect objects, review image coverage, then convert coordinates"""
        # Load the VLM once
        self.vlm.load_model()

        # Determines which coordinate instructions to use based of the model family
        uses_normalized_coordinates = self.vlm.uses_normalized_coordinates()
        prompt = self.build_prompt(width, height, uses_normalized_coordinates)

        schema = load_perception_schema()
        system_prompt = PROMPT_PATH.read_text()
        detections, replies = self.request_json(image_path, system_prompt, prompt, schema)

        # Review the full inventory before assigning IDs or computing relations
        review_prompt = (
            prompt + "\nReview this preliminary inventory against the image: "
            + json.dumps(detections)
            + "\nInspect the image region by region, including each tray rim and interior. "
            "Look for missed small or low-contrast industrial components and partly covered trays. "
            "Keep every correct detection, add only visibly present missing objects, and correct errors. "
            "Preserve existing labels and coordinates unless you can identify a specific error. "
            "Check mug handle targets too. Return the complete corrected objects list, not just additions."
        )
        detections, review_replies = self.request_json(image_path, system_prompt, review_prompt, schema)

        # Both passes use the model's coordinate convention until this final conversion
        if uses_normalized_coordinates:
            self.convert_to_pixels(detections, width, height)

        return detections, replies + review_replies

    def detect_stacking(self, image_path: Path, objects: list[dict]) -> list[dict]:
        """Ask about physical support only for overlapping non-tray object pairs"""
        candidates = stacking_candidates(objects)
        if not candidates:
            return []

        self.vlm.load_model()
        schema = load_schema("stacking.schema.json")
        candidate_ids = {object_id for pair in candidates for object_id in pair}
        candidate_objects = [obj for obj in objects if obj["id"] in candidate_ids]

        # Focus visual attention on the overlapping objects while keeping their surroundings
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read stacking image: {image_path}")
        left = min(obj["bbox_2d"][0] for obj in candidate_objects)
        top = min(obj["bbox_2d"][1] for obj in candidate_objects)
        right = max(obj["bbox_2d"][2] for obj in candidate_objects)
        bottom = max(obj["bbox_2d"][3] for obj in candidate_objects)
        padding = max(20, round(max(right - left, bottom - top) * 0.1))
        left, top = max(0, left - padding), max(0, top - padding)
        right = min(image.shape[1], right + padding)
        bottom = min(image.shape[0], bottom + padding)
        crop = image[top:bottom, left:right]

        inventory = [
            {
                "id": obj["id"], "label": obj["label"],
                "bbox_2d": [obj["bbox_2d"][0] - left, obj["bbox_2d"][1] - top,
                            obj["bbox_2d"][2] - left, obj["bbox_2d"][3] - top],
            }
            for obj in candidate_objects
        ]
        prompt = (
            f"This image is a {crop.shape[1]} by {crop.shape[0]} pixel crop. "
            "Object boxes use PIXELS IN THIS CROP, with the origin at the top left. "
            "Use the supplied IDs exactly. Inspect only the listed candidate pairs. "
            "Return only JSON matching this schema: " + json.dumps(schema)
            + "\nDetected objects: " + json.dumps(inventory)
            + "\nCandidate pairs, with no implied support direction: " + json.dumps(candidates)
        )
        with TemporaryDirectory(prefix="qwen-stacking-") as directory:
            crop_path = Path(directory) / "stacking.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError("Could not write stacking crop")
            result, _ = self.request_json(crop_path, STACKING_PROMPT_PATH.read_text(), prompt, schema)
        return [
            {"subject": support["upper"], "relation": "on",
             "object": support["lower"], "confidence": support["confidence"]}
            for support in result["supports"]
        ]

    def request_json(
        self, image_path: Path, system_prompt: str, prompt: str, schema: dict,
    ) -> tuple[dict, list[str]]:
        """Run one visual task and retry only malformed JSON responses"""

        replies = []
        errors = []

        # Include the initial attempt as well as the configured retries
        for attempt in range(self.validation_retries + 1):
            request_text = prompt

            # On a retry, tell Qwen what was wrong
            if errors:
                request_text += f"\nCorrect the JSON format errors: {errors[-1]}"

            # Prepare the system instructions, source image, and detection request
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path.resolve()),
                        },
                        {
                            "type": "text",
                            "text": request_text,
                        },
                    ],
                },
            ]

            reply = self.vlm.run_inference(messages)

            # Keep every reply so failed attempts can be inspected later
            replies.append(reply)

            try:
                # Validate the response
                result = parse_json_object(reply)
                validate_structure(result, schema, "Qwen response")
                return result, replies

            except ValueError as error:
                # Keep the error for the next attempt
                errors.append(str(error))

        raise PerceptionGenerationError(
            "Qwen did not return valid JSON for the visual task",
            replies,
            errors,
        )

    @staticmethod
    def build_prompt(
        width: int,
        height: int,
        uses_normalized_coordinates: bool,
    ) -> str:
        """Describe the coordinate system and required JSON structure."""
        # Newer Qwen models use a 0 - 1000 coordinate plane for each image axis.
        if uses_normalized_coordinates:
            coordinate_width = 1000
            coordinate_height = 1000
            coordinate_instructions = (
                "Use NORMALIZED coordinates: each axis spans 0 - 1000 across the full image. "
                "Target pixels must be in 0 - 999."
            )

        # Qwen2.5 uses original image dimensions
        else:
            coordinate_width = width
            coordinate_height = height
            coordinate_instructions = (
                f"Use original image pixels, width={width}, height={height}."
            )

        # Include the schema as text so Qwen knows which fields to return.
        schema = load_perception_schema()
        schema_text = json.dumps(schema)

        return (
            f"{coordinate_instructions} Boxes are half-open [x1,y1,x2,y2] with "
            f"0 <= x1 < x2 <= {coordinate_width}, "
            f"0 <= y1 < y2 <= {coordinate_height}. "
            "Choose target_pixel inside the object's box on visible solid material. "
            f"Return only JSON matching this schema: {schema_text}"
        )

    @staticmethod
    def convert_to_pixels(detections: dict, width: int, height: int) -> None:
        """Replace normalized box and target coordinates with source-image pixels."""
        for detection in detections["objects"]:
            x1, y1, x2, y2 = detection["bbox_2d"]

            # Scale x by image width and y by image height
            detection["bbox_2d"] = [
                math.floor(x1 * width / 1000),
                math.floor(y1 * height / 1000),
                math.ceil(x2 * width / 1000),
                math.ceil(y2 * height / 1000),
            ]

            target_x, target_y = detection["target_pixel"]
            detection["target_pixel"] = [
                math.floor(target_x * width / 1000),
                math.floor(target_y * height / 1000),
            ]
