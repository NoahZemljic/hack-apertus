import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image


LARGE_SURFACE_HINTS = ("tray", "arm", "hand")
# Only reject huge boxes for small objects implicated in the saved DINO output.
# A large plate, unknown object, or surface is not automatically a false positive.
SMALL_OBJECT_HEADS = {"gear", "cube", "mandarin", "component"}


def normalize_text(text: str) -> str:
    text = text.lower().strip(" .\t\r\n")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", normalize_text(text)).strip("_")
    if not value or not value[0].isalpha():
        value = f"object_{value}" if value else "object"
    return value


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("Expected four finite xyxy coordinates")
    x1, y1, x2, y2 = box
    x1 = min(max(float(x1), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    x2 = min(max(float(x2), 0.0), float(width))
    y2 = min(max(float(y2), 0.0), float(height))
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def box_area_ratio(box: list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / float(width * height)


def query_candidates(label: str, queries: Iterable[str]) -> list[str]:
    """Return scene queries that plausibly occur inside DINO's returned phrase.

    Grounding DINO sometimes concatenates adjacent prompt phrases, e.g.
    'mandarin red cube' or 'red gear yellow gear'. We keep every matching
    original query so later logic can resolve the phrase conservatively.
    """
    label_n = normalize_text(label)
    queries = list(dict.fromkeys(normalize_text(q) for q in queries if normalize_text(q)))
    # 'green mug' and 'red gear' are already valid names, even when the prompt
    # also contains 'mug' or 'gear'. Never expand their candidate vocabulary.
    if label_n in queries:
        return [label_n]
    spans = {
        q: [match.span() for match in re.finditer(rf"(?<!\S){re.escape(q)}(?!\S)", label_n)]
        for q in queries
    }
    # Ignore generic words only at positions covered by a longer query. In
    # 'plate yellow plate', the first 'plate' is still a separate candidate.
    return [q for q, matches in spans.items() if any(
        not any(
            other != q and other_start <= start and end <= other_end
            for other, other_matches in spans.items()
            for other_start, other_end in other_matches
        )
        for start, end in matches
    )]


def cluster_by_iou(detections: list[dict], threshold: float) -> list[list[dict]]:
    """Group boxes against the strongest box, avoiding transitive IoU chains.

    Some detectors do not supply confidence. Their boxes retain input order;
    no confidence is manufactured for duplicate selection.
    """
    remaining = sorted(detections, key=score_priority, reverse=True)
    clusters = []
    while remaining:
        representative = remaining.pop(0)
        cluster = [representative]
        separate = []
        for detection in remaining:
            if box_iou(representative["box"], detection["box"]) >= threshold:
                cluster.append(detection)
            else:
                separate.append(detection)
        clusters.append(cluster)
        remaining = separate
    return clusters


def score_priority(detection: dict) -> tuple[bool, float]:
    score = detection["score"]
    return score is not None, float(score) if score is not None else 0.0


def resolve_label(label: str, candidates: list[str]) -> str:
    """Only collapse repeated copies of one complete query; never guess names.

    Exact labels and unknown names retain their original spelling. Compound
    phrases containing different queries require review, not scene-count or
    phrase-length heuristics.
    """
    if len(candidates) == 1:
        query = candidates[0]
        if re.fullmatch(rf"{re.escape(query)}(?: {re.escape(query)})+", normalize_text(label)):
            return query
    return label


def assign_ids(detections: list[dict]) -> list[dict]:
    counts = Counter()
    result = []
    for detection in detections:
        stem = slug(detection["label"])
        counts[stem] += 1
        result.append({**detection, "id": f"{stem}_{counts[stem]}"})
    return result


def process_scene(
    item: dict,
    image_path: Path,
    duplicate_iou: float = 0.90,
    max_object_area_ratio: float = 0.35,
) -> tuple[dict, dict]:
    if not 0 < duplicate_iou <= 1:
        raise ValueError("duplicate_iou must be in (0, 1]")
    if not 0 < max_object_area_ratio <= 1:
        raise ValueError("max_object_area_ratio must be in (0, 1]")
    with Image.open(image_path) as image:
        width, height = image.size

    queries = [normalize_text(q) for q in item.get("queries", [])]
    prepared = []
    dropped = []
    clipped_count = 0

    for raw in item.get("detections", []):
        d = dict(raw)
        score = d.get("score")
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        ):
            raise ValueError(f"{item['scene']}: detection score must be a finite number or null")
        d["score"] = score
        if "backend" not in d and "backend" in item:
            d["backend"] = item["backend"]
        try:
            d["box"] = clip_box(d["box"], width, height)
        except (TypeError, ValueError, OverflowError):
            dropped.append({**d, "drop_reason": "invalid_box"})
            continue
        x1, y1, x2, y2 = d["box"]
        if x2 <= x1 or y2 <= y1:
            dropped.append({**d, "drop_reason": "invalid_box"})
            continue
        clipped_count += d["box"] != raw["box"]

        d["area_ratio"] = round(box_area_ratio(d["box"], width, height), 5)
        d["query_candidates"] = query_candidates(d["label"], queries)

        label_n = normalize_text(d["label"])
        is_large_surface = any(hint in label_n.split() for hint in LARGE_SURFACE_HINTS)
        candidate_large_surface = any(
            any(hint in candidate.split() for hint in LARGE_SURFACE_HINTS)
            for candidate in d["query_candidates"]
        )
        is_small_object = bool(label_n) and label_n.split()[-1] in SMALL_OBJECT_HEADS

        if (
            d["area_ratio"] > max_object_area_ratio
            and is_small_object
            and not is_large_surface
            and not candidate_large_surface
        ):
            dropped.append({**d, "drop_reason": "pathologically_large_box"})
            continue

        d["oversized_needs_review"] = (
            d["area_ratio"] > max_object_area_ratio
            and not is_large_surface
            and not candidate_large_surface
        )
        prepared.append(d)

    clusters = cluster_by_iou(prepared, duplicate_iou)

    cleaned = []
    for cluster in clusters:
        representative = cluster[0]
        aliases = sorted({q for d in cluster for q in d["query_candidates"]})
        canonical = resolve_label(representative["label"], representative["query_candidates"])
        raw_labels = list(dict.fromkeys(d["label"] for d in cluster))
        review_reasons = []
        if len(representative["query_candidates"]) > 1:
            review_reasons.append("ambiguous_compound_label")
        if len({normalize_text(label) for label in raw_labels}) > 1:
            review_reasons.append("conflicting_duplicate_labels")
        if representative["oversized_needs_review"]:
            review_reasons.append("oversized_box")
        for duplicate in cluster:
            if duplicate is not representative:
                dropped.append({
                    **duplicate,
                    "drop_reason": "duplicate_box",
                    "kept_source_id": representative.get("id"),
                })
        cleaned.append(
            {
                "label": canonical,
                "aliases": aliases,
                "raw_labels": raw_labels,
                "source_ids": [d["id"] for d in cluster if "id" in d],
                "box": representative["box"],
                "score": round(float(representative["score"]), 4) if representative["score"] is not None else None,
                **({"backend": representative["backend"]} if "backend" in representative else {}),
                **({"detector_score": representative["detector_score"]} if "detector_score" in representative else {}),
                "area_ratio": representative["area_ratio"],
                "merged_count": len(cluster),
                "needs_review": bool(review_reasons),
                "review_reasons": review_reasons,
            }
        )

    # Stable spatial order makes label-based IDs reproducible across runs.
    cleaned.sort(key=lambda d: (d["box"][1], d["box"][0], d["label"]))
    cleaned = assign_ids(cleaned)

    stats = {
        "scene": item["scene"],
        "raw_count": len(item.get("detections", [])),
        "clean_count": len(cleaned),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "merged_count": sum(d["merged_count"] - 1 for d in cleaned),
        "clipped_count": clipped_count,
        "review_ids": [d["id"] for d in cleaned if d["needs_review"]],
    }

    return {
        "scene": item["scene"],
        **({"backend": item["backend"]} if "backend" in item else {}),
        "queries": item.get("queries", []),
        "detections": cleaned,
    }, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative Grounding DINO post-processing")
    parser.add_argument("--input", type=Path, default=Path("outputs/grounded_pipeline/grounding_detections.json"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grounded_pipeline/grounding_detections_clean.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/grounded_pipeline/grounding_cleanup_report.json"))
    parser.add_argument("--duplicate-iou", type=float, default=0.90)
    parser.add_argument("--max-object-area-ratio", type=float, default=0.35)
    args = parser.parse_args()

    data = json.loads(args.input.read_text().replace("\u00a0", " "))
    cleaned_all = []
    reports = []

    for item in data:
        image_path = args.dataset / item["scene"] / "rgb.png"
        cleaned, report = process_scene(
            item,
            image_path,
            duplicate_iou=args.duplicate_iou,
            max_object_area_ratio=args.max_object_area_ratio,
        )
        cleaned_all.append(cleaned)
        reports.append(report)

        print(
            f'{item["scene"]}: '
            f'{report["raw_count"]} raw -> {report["clean_count"]} clean '
            f'({report["dropped_count"]} dropped)'
        )

    for path, results in [(args.output, cleaned_all), (args.report, reports)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved cleaned detections to {args.output}")
    print(f"Saved cleanup report to {args.report}")


if __name__ == "__main__":
    main()
