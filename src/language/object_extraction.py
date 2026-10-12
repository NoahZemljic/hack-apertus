import argparse
import json
import urllib.request
from pathlib import Path
from src.schemas import load_object_extraction_schema


BASE_URL = "http://localhost:8000/v1"
MODEL = "swiss-ai/Apertus-v1.5-8B"

PROMPT = """
Extract every explicitly mentioned concrete object or object category from the instruction.

Rules:
- Preserve the full identifying phrase exactly as stated, including descriptive words such as color, size, material, or other attributes.
- Do not infer or add information that is not explicitly stated.
- Do not replace a specific object description with a more general category.
- Do not extract generic references such as "object", "thing", "one", "that one", "it", "everything", or similar referential words.
- If the target itself is described only by a generic reference, still extract any concrete reference objects mentioned in the relation.
- Broad but meaningful object categories should still be extracted when they are explicitly stated and no more specific description is given.
- If multiple different object descriptions are explicitly mentioned, return each separately.
- If the same object description occurs more than once, return it only once.
""".strip()


def load_instructions(dataset_path: Path) -> list[dict]:
    instructions = []

    for scene_dir in sorted(dataset_path.glob("scene_*")):
        instructions_file = scene_dir / "instructions.txt"

        if not instructions_file.exists():
            continue

        for line_number, line in enumerate(
            instructions_file.read_text(encoding="utf-8").splitlines(),
            start=1
        ):
            line = line.strip()

            # Ignore comments and empty lines
            if not line or line.startswith("#"):
                continue

            category, instruction = line.split("\t", 1)

            instructions.append({
                "scene": scene_dir.name,
                "category": category,
                "instruction": instruction
            })

    return instructions


def ask_apertus(
    instruction: str, schema: dict, *, base_url: str = BASE_URL,
    model: str = MODEL, timeout: float = 60, constrained: bool = True,
) -> dict:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": f"{PROMPT}\n\nInstruction: {instruction}"
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "object_extraction",
                "schema": schema
            }
        }
    }

    if not constrained:
        body.pop("response_format")

    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer none"
        }
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())

    extraction = json.loads(
        result["choices"][0]["message"]["content"]
    )
    if (not isinstance(extraction, dict) or set(extraction) != {"objects"}
        or not isinstance(extraction["objects"], list)
        or any(not isinstance(obj, str) or not obj.strip() for obj in extraction["objects"])):
        raise ValueError("Apertus extraction must contain only an objects list of nonempty strings")
    return extraction


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset")
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schema/object_extraction.schema.json")
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/grounded_pipeline/object_extractions.json")
    )

    args = parser.parse_args()

    schema = load_object_extraction_schema()
    instructions = load_instructions(args.data)

    results = []

    for i, item in enumerate(instructions, start=1):
        instruction = item["instruction"]

        extraction = ask_apertus(
            instruction=instruction,
            schema=schema
        )

        result = {
            "scene": item["scene"],
            "category": item["category"],
            "original_instruction": instruction,
            "extracted_objects": extraction["objects"]
        }

        results.append(result)

        print(f"\n[{i}/{len(instructions)}] {item['scene']}")
        print(f"Category:    {item['category']}")
        print(f"Instruction: {instruction}")
        print(f"Objects:     {extraction['objects']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
