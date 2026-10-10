import json

def build_unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys rather than silently replacing earlier evidence"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}; include each field exactly once")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    """Reject non-JSON numeric constants such as NaN and Infinity"""
    raise ValueError(f"Invalid JSON number {value}; use a finite numeric value")


def parse_json_object(text: str) -> dict:
    """Parse exactly one complete JSON object without duplicate fields or nonstandard numbers"""
    value = json.loads(text, object_pairs_hook=build_unique_object, parse_constant=reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object, not an array, scalar, or null")
    return value
