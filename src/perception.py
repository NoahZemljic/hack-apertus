import json
import math
from pathlib import Path
from typing import Protocol

from .json_tools import parse_json_object
from .schemas import load_perception_schema
from .validation import validate_perception


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "qwen_system.txt"

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
        """Run detection, retry format errors, and return the accepted response."""
        # Load the VLM once
        self.vlm.load_model()

        # Determines which coordinate instructions to use based of the model family
        uses_normalized_coordinates = self.vlm.uses_normalized_coordinates()
        prompt = self.build_prompt(width, height, uses_normalized_coordinates)

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
                    "content": PROMPT_PATH.read_text(),
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

            # Keep every reply so failed attempts can be inspected later.
            replies.append(reply)

            try:
                # Validate the response
                detections = parse_json_object(reply)
                validate_perception(detections)

                # Convert normalized coordinates before returning to the pipeline
                if uses_normalized_coordinates:
                    self.convert_to_pixels(detections, width, height)

                return detections, replies

            except ValueError as error:
                # Keep the error for the next attempt
                errors.append(str(error))

        raise PerceptionGenerationError(
            "Qwen did not return valid detection JSON",
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
