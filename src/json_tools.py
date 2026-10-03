import json


def extract_json_object(text: str) -> dict:
    """Extract the first complete JSON object from a model response"""
    cleaned = text.replace("```json", "").replace("```", "")
    decoder = json.JSONDecoder()

    # Models may add prose, so try each opening brace until valid JSON begins
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response does not contain a JSON object")
