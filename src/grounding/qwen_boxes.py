"""One Qwen inventory-and-box request, adapted to the shared SAM input."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.grounding.detections import normalize_queries
from src.grounding.postprocess_grounding import assign_ids, normalize_text
from src.json_tools import parse_json_object
from src.scene_discovery import image_to_data_url
from src.schemas import load_schema
from src.validation import validate_structure


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "qwen_boxes_system.txt"


class QwenBoxGenerationError(ValueError):
    """Retain the actual model replies and validation errors on exhaustion."""

    def __init__(self, message: str, replies: list[str], errors: list[str]) -> None:
        super().__init__(message)
        self.replies = list(replies)
        self.errors = list(errors)


class QwenBoxDetector:
    """Use an existing Qwen HTTP server; retry only malformed detector output."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-32B-Instruct",
        base_url: str = "http://127.0.0.1:8001/v1",
        timeout: float = 180,
        constrained: bool = True,
        retries: int = 1,
    ) -> None:
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("Qwen box retries must be a nonnegative integer")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.constrained = constrained
        self.retries = retries
        self.last_trace: list[dict] = []
        self.replies: list[str] = []
        self.errors: list[str] = []

    def detect(self, image_path: Path, hints: Iterable[str]) -> list[dict]:
        """Return independent instances in source-image, half-open xyxy pixels.

        Qwen is asked to discover beyond hints. Unknown confidence is represented
        by None, rather than a fabricated DINO probability. A valid empty inventory
        succeeds; neither missing categories nor scene semantics trigger a retry.
        """
        self.last_trace = []
        self.replies = []
        self.errors = []
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        schema = load_schema("qwen_boxes.schema.json")
        system_prompt = PROMPT_PATH.read_text().strip()
        hint_terms = sorted({hint.strip().lower() for hint in hints if hint.strip()})
        prompt = (
            f"Inspect this full {width} by {height} pixel image. Return all visible relevant "
            "object instances, including objects outside the attention hints. "
            "Use NORMALIZED coordinates with each full-image axis spanning 0 to 1000, "
            "not original-image pixels. Boxes are half-open [x_min,y_min,x_max,y_max] "
            "with 0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000. "
            "Attention hints: " + json.dumps(hint_terms)
            + "\nReturn only JSON matching this schema: " + json.dumps(schema)
        )
        image_url = image_to_data_url(image_path)

        for attempt in range(self.retries + 1):
            request_prompt = prompt
            if self.errors:
                request_prompt += (
                    "\nCorrect the JSON format or coordinate error in your previous "
                    f"response: {self.errors[-1]}. Return the complete objects JSON."
                )
            body = {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": request_prompt},
                    ]},
                ],
            }
            if self.constrained:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qwen_boxes", "strict": True, "schema": schema,
                    },
                }
            # Keep prompts and replies, without duplicating the full encoded image
            # into each trace entry and run artifact.
            trace = {
                "stage": "qwen_boxes", "attempt": attempt + 1,
                "model": self.model, "image_path": str(image_path),
                "image_size": {"width": width, "height": height},
                "coordinate_scale": "normalized_0_1000_half_open",
                "system_prompt": system_prompt, "prompt": request_prompt,
                "temperature": 0, "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
                "constrained_decoding": self.constrained,
            }
            self.last_trace.append(trace)
            request = urllib.request.Request(
                self.base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                trace["request_error"] = f"{type(error).__name__}: {error}"
                raise
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                trace["request_error"] = "Qwen server returned an invalid JSON response envelope"
                raise ValueError(trace["request_error"]) from error
            try:
                reply = payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                # A malformed HTTP envelope is a server failure, not a model
                # formatting error and must not turn into a fake empty inventory.
                trace["request_error"] = "Qwen server returned no completion content"
                raise ValueError(trace["request_error"]) from error
            self.replies.append(reply)
            trace["reply"] = reply
            try:
                if not isinstance(reply, str):
                    raise ValueError("Qwen response content must be a JSON string")
                result = validate_structure(parse_json_object(reply), schema, "Qwen boxes")
                detections = self._to_detections(result, width, height)
            except ValueError as error:
                self.errors.append(str(error))
                trace["validation_error"] = str(error)
                continue
            trace["validation_error"] = None
            return detections

        raise QwenBoxGenerationError(
            f"Qwen did not return valid object boxes: {self.errors[-1]}",
            self.replies, self.errors,
        )

    @staticmethod
    def _to_detections(result: dict, width: int, height: int) -> list[dict]:
        detections = []
        for index, obj in enumerate(result["objects"]):
            x1, y1, x2, y2 = obj["bbox_2d"]
            if any(type(value) is not int for value in obj["bbox_2d"]):
                raise ValueError(f"objects[{index}].bbox_2d must contain integer coordinates")
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError(
                    f"objects[{index}].bbox_2d must be ordered, positive-area "
                    "normalized bounds within 0 to 1000"
                )
            labels = normalize_queries([normalize_text(obj["label"])])
            if not labels:
                raise ValueError(f"objects[{index}].label must contain an object name")
            label = labels[0]
            detections.append({
                "label": label, "query": label,
                "box": [math.floor(x1 * width / 1000), math.floor(y1 * height / 1000),
                        math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000)],
                "score": None, "backend": "qwen",
            })
        return assign_ids(detections)
