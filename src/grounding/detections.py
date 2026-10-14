"""Shared category normalization and raw detection identifiers."""

import re
from collections import Counter


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

