"""Image encoding and object hints shared by Qwen perception requests."""

import base64
from pathlib import Path


GENERIC_REFERENCES = {
    "object",
    "thing",
    "one",
    "that one",
    "it",
    "everything",
}


def image_to_data_url(image_path: Path) -> str:
    image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:image/png;base64,{image}"


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
