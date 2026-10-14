from __future__ import annotations

import base64
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from src.prompting import (
    prepare_scene_discovery_prompt,
    prepare_verification_prompt,
)
from src.schemas import load_scene_discovery_schema


OBJECT_KEYWORDS = Path("outputs/grounded_pipeline/object_extractions.json")
DATASET = Path("dataset")
OUTPUT = Path("outputs/grounded_pipeline/scene_discovery.json")

QWEN_URL = "http://127.0.0.1:8001/v1"
QWEN_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


GENERIC_REFERENCES = {
    "object",
    "thing",
    "one",
    "that one",
    "it",
    "everything",
}


def load_object_extraction_keywords(
    path: Path,
) -> dict[str, list[str]]:
    """Load Apertus-extracted object terms by scene."""

    scene_list = json.loads(path.read_text())

    keywords = defaultdict(set)

    for scene_item in scene_list:
        scene = scene_item["scene"]

        for obj in scene_item["extracted_objects"]:
            obj = obj.strip().lower()

            if obj and obj not in GENERIC_REFERENCES:
                keywords[scene].add(obj)

    return {
        scene: sorted(objects)
        for scene, objects in keywords.items()
    }


def image_to_data_url(image_path: Path) -> str:
    image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:image/png;base64,{image}"


def build_query(
    prompt: str,
    schema: dict,
    image_path: Path,
    *, model: str = QWEN_MODEL,
) -> dict[str, Any]:

    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 800,

        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path)
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],

        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "scene_discovery",
                "schema": schema,
            },
        },
    }


def call_qwen(
    client: OpenAI | None,
    query: dict[str, Any],
    *, base_url: str = QWEN_URL, timeout: float = 180,
) -> list[str]:
    """Send one request to Qwen."""

    if client is None:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(query).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = json.loads(response.read())["choices"][0]["message"]["content"]
    else:
        response = client.chat.completions.create(**query)
        text = response.choices[0].message.content
    result = json.loads(text)
    if (not isinstance(result, dict) or set(result) != {"objects"}
        or not isinstance(result["objects"], list) or not result["objects"]
        or any(not isinstance(obj, str) or not obj.strip() or len(obj) > 80 for obj in result["objects"])):
        raise ValueError("Qwen discovery must contain only a nonempty objects list of short labels")
    return result["objects"]


def clean_phrases(
    phrases: list[str],
) -> list[str]:
    """
    Only perform trivial cleanup.

    No semantic normalization.
    """

    return sorted({
        phrase.strip().lower()
        for phrase in phrases
        if phrase.strip()
    })


def build_grounding_queries(
    instruction_objects: list[str],
    qwen_objects: list[str],
) -> list[str]:
    """
    Combine Apertus instruction extraction with Qwen visual discovery.

    Both sources are useful:
    - Apertus gives task-relevant terminology.
    - Qwen can discover additional visible objects.
    """

    queries = (
        instruction_objects
        + qwen_objects
    )

    return clean_phrases(queries)


def discover_scene(
    client: OpenAI | None,
    image_path: Path,
    schema: dict,
    hints: list[str],
    *, model: str = QWEN_MODEL, base_url: str = QWEN_URL,
    timeout: float = 180, constrained: bool = True,
) -> list[str]:
    """
    Run Qwen discovery followed by one verification pass.
    """

    discovery_prompt = (
        prepare_scene_discovery_prompt(hints)
    )

    discovery_query = build_query(discovery_prompt, schema, image_path, model=model)
    if not constrained:
        discovery_query.pop("response_format")
    discovered = call_qwen(
        client,
        discovery_query, base_url=base_url, timeout=timeout,
    )

    discovered = clean_phrases(discovered)

    print(
        "Qwen discovery:",
        discovered,
    )

    verification_prompt = (
        prepare_verification_prompt(
            hints,
            discovered,
        )
    )

    verification_query = build_query(verification_prompt, schema, image_path, model=model)
    if not constrained:
        verification_query.pop("response_format")
    verified = call_qwen(
        client,
        verification_query, base_url=base_url, timeout=timeout,
    )

    verified = clean_phrases(verified)

    print(
        "Qwen verified:",
        verified,
    )

    return verified


def write_results(
    results: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def main():
    from openai import OpenAI

    instruction_objects = (
        load_object_extraction_keywords(
            OBJECT_KEYWORDS
        )
    )

    schema = load_scene_discovery_schema()

    client = OpenAI(
        base_url=QWEN_URL,
        api_key="EMPTY",
    )

    results = []

    for scene, hints in sorted(
        instruction_objects.items()
    ):

        print()
        print("=" * 60)
        print(f"Processing {scene}")

        image_path = (
            DATASET
            / scene
            / "rgb.png"
        )

        qwen_objects = discover_scene(
            client=client,
            image_path=image_path,
            schema=schema,
            hints=hints,
        )

        grounding_queries = (
            build_grounding_queries(
                instruction_objects=hints,
                qwen_objects=qwen_objects,
            )
        )

        print(
            "Grounding DINO queries:",
            grounding_queries,
        )

        results.append(
            {
                "scene": scene,
                "instruction_objects": hints,
                "qwen_objects": qwen_objects,
                "grounding_queries": grounding_queries,
            }
        )

    write_results(
        results,
        OUTPUT,
    )


if __name__ == "__main__":
    main()
