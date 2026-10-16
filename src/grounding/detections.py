"""Shared category normalization for Qwen object labels."""


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
