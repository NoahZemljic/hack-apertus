"""One separate visual task for direct support between existing objects."""

from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
import re
import urllib.error
import urllib.request

from PIL import Image

from src.json_tools import parse_json_object
from src.scene_discovery import image_to_data_url
from src.schemas import load_schema
from src.validation import validate_structure


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "qwen_stacking_system.txt"
EXCLUDED_WORDS = {
    "tray", "trays", "container", "containers", "bin", "bins", "basket", "baskets",
    "hand", "hands", "arm", "arms", "human", "humans", "person", "people",
}


def _objects_by_id(objects: list[dict]) -> dict[str, dict]:
    inventory = {}
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Stacking objects must be dictionaries")
        object_id, label, box = obj.get("id"), obj.get("label"), obj.get("box")
        if not isinstance(object_id, str) or not object_id.strip() or object_id in inventory:
            raise ValueError("Stacking objects need distinct, nonempty IDs")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{object_id}: stacking label must be nonempty")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
            raise ValueError(f"{object_id}: stacking box must contain finite, positive-area pixel bounds")
        inventory[object_id] = obj
    return inventory


def stacking_candidates(objects: list[dict]) -> list[tuple[str, str]]:
    """Locate possible contacts; overlapping boxes never establish support.

    Original detector boxes retain partially occluded object extents better than
    independently segmented masks. A small tolerance also includes touching or
    nearly touching visible extents, without assigning either support direction.
    Container membership and human contact are separate scene graph properties.
    """
    _objects_by_id(objects)
    eligible = [obj for obj in objects if not EXCLUDED_WORDS.intersection(
        re.findall(r"[a-z]+", obj["label"].lower())
    )]
    candidates = []
    for first, second in combinations(eligible, 2):
        a, b = first["box"], second["box"]
        shorter_extent = min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
        tolerance = min(12.0, max(2.0, shorter_extent * 0.03))
        if (min(a[2], b[2]) + tolerance >= max(a[0], b[0])
                and min(a[3], b[3]) + tolerance >= max(a[1], b[1])):
            candidates.append((first["id"], second["id"]))
    return candidates


class QwenStackingGenerationError(ValueError):
    """Retain model replies and contract errors when the retry allowance ends."""

    def __init__(self, message: str, replies: list[str], errors: list[str]) -> None:
        super().__init__(message)
        self.replies = list(replies)
        self.errors = list(errors)


class QwenStackingDetector:
    """Ask Qwen only for direct support, without revising the detected inventory."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-32B-Instruct",
        base_url: str = "http://127.0.0.1:8001/v1",
        timeout: float = 180,
        constrained: bool = True,
        retries: int = 1,
    ) -> None:
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("Qwen stacking retries must be a nonnegative integer")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.constrained = constrained
        self.retries = retries
        self.last_trace: list[dict] = []
        self.replies: list[str] = []
        self.errors: list[str] = []

    def detect(self, image_path: Path, objects: list[dict]) -> list[dict]:
        """Return validated direct {upper, lower} links between supplied IDs.

        There is one full-image task for all candidate pairs. Only malformed
        output gets a retry; an empty support list is a successful visual answer.
        Scenes with no candidates do not make a visual request at all.
        """
        self.last_trace = []
        self.replies = []
        self.errors = []
        candidates = stacking_candidates(objects)
        if not candidates:
            return []
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        candidate_ids = {object_id for pair in candidates for object_id in pair}
        inventory = [
            {"id": obj["id"], "label": obj["label"], "bbox_2d": list(obj["box"])}
            for obj in objects if obj["id"] in candidate_ids
        ]
        if any(obj["bbox_2d"][2] > width or obj["bbox_2d"][3] > height for obj in inventory):
            raise ValueError("Stacking object boxes must lie within the source image")
        schema = load_schema("qwen_stacking.schema.json")
        system_prompt = PROMPT_PATH.read_text().strip()
        prompt = (
            f"Inspect this full {width} by {height} pixel image for direct physical support. "
            "The supplied boxes use ORIGINAL FULL-IMAGE PIXELS, with the origin at the top left. "
            "They only identify existing objects; do not return coordinates or change any detection. "
            "Use the supplied IDs exactly. Inspect only the listed candidate pairs, considering "
            "both possible support directions. Return only JSON matching this schema: "
            + json.dumps(schema)
            + "\nDetected candidate objects: " + json.dumps(inventory)
            + "\nCandidate pairs, with no implied support direction: " + json.dumps(candidates)
        )
        image_url = image_to_data_url(image_path)
        for attempt in range(self.retries + 1):
            request_prompt = prompt
            if self.errors:
                request_prompt += (
                    "\nCorrect the JSON or relationship-contract error in your previous response: "
                    f"{self.errors[-1]}. Return the complete supports JSON using only candidate IDs."
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
                    "json_schema": {"name": "qwen_stacking", "strict": True, "schema": schema},
                }
            trace = {
                "stage": "qwen_stacking", "attempt": attempt + 1,
                "model": self.model, "image_path": str(image_path),
                "image_size": {"width": width, "height": height},
                "coordinate_scale": "original_image_pixels_half_open",
                "system_prompt": system_prompt, "prompt": request_prompt,
                "objects": inventory, "candidate_pairs": candidates,
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
            except urllib.error.HTTPError as error:
                # Keep the server's actual rejection reason. A 400 contains no
                # model answer, so it must not consume a format retry or cause
                # another visual request with different decoding settings.
                try:
                    error_body = error.read(16384).decode("utf-8", errors="replace")
                except (OSError, ValueError, AttributeError):
                    error_body = ""
                trace["http_status"] = error.code
                trace["http_error_body"] = error_body
                if error_body:
                    error.msg = f"{error.msg}; server response: {error_body}"
                trace["request_error"] = f"{type(error).__name__}: {error}"
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                trace["request_error"] = f"{type(error).__name__}: {error}"
                raise
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                trace["request_error"] = "Qwen server returned an invalid JSON response envelope"
                raise ValueError(trace["request_error"]) from error
            try:
                reply = payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                trace["request_error"] = "Qwen server returned no completion content"
                raise ValueError(trace["request_error"]) from error
            self.replies.append(reply)
            trace["reply"] = reply
            try:
                if not isinstance(reply, str):
                    raise ValueError("Qwen response content must be a JSON string")
                result = validate_structure(parse_json_object(reply), schema, "Qwen stacking")
                supports = self._validate_supports(result["supports"], objects, candidates)
            except ValueError as error:
                self.errors.append(str(error))
                trace["validation_error"] = str(error)
                continue
            trace["validation_error"] = None
            return supports

        raise QwenStackingGenerationError(
            f"Qwen did not return valid stacking relationships: {self.errors[-1]}",
            self.replies, self.errors,
        )

    @staticmethod
    def _validate_supports(supports: list[dict], objects: list[dict],
                           candidates: list[tuple[str, str]]) -> list[dict]:
        known_ids = {obj["id"] for obj in objects}
        candidate_pairs = {frozenset(pair) for pair in candidates}
        edges = set()
        for support in supports:
            upper, lower = support["upper"], support["lower"]
            if upper not in known_ids or lower not in known_ids:
                raise ValueError(f"Stacking relationship references an unknown object: {upper}, {lower}")
            if upper == lower:
                raise ValueError(f"Stacking relationship cannot reference itself: {upper}")
            if frozenset((upper, lower)) not in candidate_pairs:
                raise ValueError(f"Stacking relationship is not a candidate pair: {upper}, {lower}")
            if (upper, lower) in edges:
                raise ValueError(f"Duplicate stacking relationship: {upper}, {lower}")
            if (lower, upper) in edges:
                raise ValueError(f"Opposing stacking relationships: {upper}, {lower}")
            edges.add((upper, lower))

        # Topological ordering permits many direct supports but rejects cycles.
        children = {object_id: [] for object_id in known_ids}
        indegree = dict.fromkeys(known_ids, 0)
        for upper, lower in edges:
            children[upper].append(lower)
            indegree[lower] += 1
        ready = [object_id for object_id, degree in indegree.items() if not degree]
        visited = 0
        while ready:
            upper = ready.pop()
            visited += 1
            for lower in children[upper]:
                indegree[lower] -= 1
                if not indegree[lower]:
                    ready.append(lower)
        if visited != len(known_ids):
            raise ValueError("Stacking relationships must not form a cycle")
        return [dict(support) for support in supports]
