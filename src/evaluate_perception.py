"""
Evaluate saved Qwen bounding boxes and stacking relations without loading models.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = (0.5, 0.75)
Box = tuple[float, float, float, float]
AnnotationId = int | str


def parse_box(values: object, context: str, *, coco: bool = False) -> Box:
    """Validate a box and convert COCO xywh to xyxy when requested."""
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{context}: expected four bounding-box coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError(f"{context}: bounding-box coordinates must be numbers")
    x1, y1, x2, y2 = map(float, values)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"{context}: bounding-box coordinates must be finite")
    if coco:
        if x2 < 0 or y2 < 0:
            raise ValueError(f"{context}: COCO width and height must be nonnegative")
        x2, y2 = x1 + x2, y1 + y2
    if not all(math.isfinite(v) for v in (x2, y2)) or x2 < x1 or y2 < y1:
        raise ValueError(f"{context}: invalid xyxy bounding box")
    return x1, y1, x2, y2


def load_ground_truth(path: Path) -> dict[str, list[tuple[AnnotationId, Box]]]:
    """Group COCO annotation IDs and boxes by scene filename."""
    data = json.loads(path.read_text(encoding="utf-8"))
    scenes: dict[str, list[tuple[AnnotationId, Box]]] = {}
    image_scenes = {}
    for image in data["images"]:
        scene = Path(image["file_name"]).stem
        if scene in scenes or image["id"] in image_scenes:
            raise ValueError(f"{path}: duplicate scene name or image ID: {scene}")
        scenes[scene] = []
        image_scenes[image["id"]] = scene
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in image_scenes:
            raise ValueError(f"{path}: annotation references unknown image ID {image_id}")
        scene = image_scenes[image_id]
        annotation_id = annotation["id"]
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, (int, str)):
            raise ValueError(f"{path}: invalid annotation ID in {scene}")
        if any(existing_id == annotation_id for existing_id, _ in scenes[scene]):
            raise ValueError(f"{path}: duplicate annotation ID {annotation_id} in {scene}")
        box = parse_box(annotation["bbox"], f"{path}: {scene}", coco=True)
        scenes[scene].append((annotation_id, box))
    return scenes


def load_predictions(path: Path, scene: str) -> tuple[list[tuple[str, Box]], int]:
    """Read original boxes and their IDs, preserving their matching order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "scene" in data and data["scene"] != scene:
        raise ValueError(f"{path}: scene does not match the prediction filename")
    objects = data["objects"]
    if not isinstance(objects, list):
        raise ValueError(f"{path}: objects must be a list")
    boxes = []
    seen_ids = set()
    missing = 0
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: object {index} must be an object")
        object_id = obj.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{path}: object {index} needs an ID for stacking evaluation")
        if object_id in seen_ids:
            raise ValueError(f"{path}: duplicate object ID {object_id}")
        seen_ids.add(object_id)
        values = obj.get("box")
        if values is None:
            missing += 1
        else:
            boxes.append((object_id, parse_box(values, f"{path}: object {index} box")))
    return boxes, missing


def load_stacking_ground_truth(path: Path) -> dict[str, set[tuple[AnnotationId, AnnotationId]]]:
    """Read directed (upper, lower) relations between annotation IDs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected stacking annotations by scene")
    scenes = {}
    for scene, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("on_top_of"), list):
            raise ValueError(f"{path}: {scene} needs an on_top_of list")
        edges = set()
        for relation in entry["on_top_of"]:
            if not isinstance(relation, dict):
                raise ValueError(f"{path}: invalid stacking relation in {scene}")
            upper, lower = relation["upper"], relation["lower"]
            if any(isinstance(v, bool) or not isinstance(v, (int, str)) for v in (upper, lower)):
                raise ValueError(f"{path}: invalid annotation ID in {scene}")
            edges.add((upper, lower))
        scenes[scene] = edges
    return scenes


def load_predicted_stacking(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Read directed on_top_of relations from the planner scene graph."""
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not isinstance(graph.get("objects"), list):
        raise ValueError(f"{path}: objects must be a list")
    object_ids = set()
    edges = set()
    for obj in graph["objects"]:
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or not obj["id"]:
            raise ValueError(f"{path}: each object needs a nonempty ID")
        object_id = obj["id"]
        if object_id in object_ids:
            raise ValueError(f"{path}: duplicate object ID {object_id}")
        object_ids.add(object_id)
        relations = obj.get("relations", {})
        if not isinstance(relations, dict):
            raise ValueError(f"{path}: invalid relations for {object_id}")
        supports = relations.get("on_top_of", [])
        if supports is None:
            supports = []
        elif isinstance(supports, str):
            supports = [supports]
        if not isinstance(supports, list) or any(not isinstance(v, str) or not v for v in supports):
            raise ValueError(f"{path}: invalid on_top_of for {object_id}")
        edges.update((object_id, lower) for lower in supports)
    return object_ids, edges


def pairwise_iou(predictions: list[Box], ground_truth: list[Box]) -> list[list[float]]:
    """Return the prediction-by-ground-truth IoU matrix for xyxy boxes."""
    matrix = []
    for x1, y1, x2, y2 in predictions:
        row = []
        pred_area = (x2 - x1) * (y2 - y1)
        for gx1, gy1, gx2, gy2 in ground_truth:
            intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(
                0.0, min(y2, gy2) - max(y1, gy1)
            )
            union = pred_area + (gx2 - gx1) * (gy2 - gy1) - intersection
            row.append(intersection / union if union > 0 else 0.0)
        matrix.append(row)
    return matrix


def hungarian_maximize(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-weight rectangular assignment using the Hungarian algorithm."""
    if not weights or not weights[0]:
        return []
    transposed = len(weights) > len(weights[0])
    if transposed:
        weights = [list(row) for row in zip(*weights)]
    n, m = len(weights), len(weights[0])  # n <= m

    # Potentials and predecessor links for successive augmenting paths.
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    assigned_row, previous = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        assigned_row[0] = row
        column = 0
        min_cost = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[column] = True
            current_row = assigned_row[column]
            delta, next_column = math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cost = -weights[current_row - 1][j - 1] - u[current_row] - v[j]
                    if cost < min_cost[j]:
                        min_cost[j], previous[j] = cost, column
                    if min_cost[j] < delta:
                        delta, next_column = min_cost[j], j
            for j in range(m + 1):
                if used[j]:
                    u[assigned_row[j]] += delta
                    v[j] -= delta
                else:
                    min_cost[j] -= delta
            column = next_column
            if assigned_row[column] == 0:
                break
        while column:
            prior_column = previous[column]
            assigned_row[column] = assigned_row[prior_column]
            column = prior_column

    pairs = [(assigned_row[j] - 1, j - 1) for j in range(1, m + 1) if assigned_row[j]]
    return [(column, row) for row, column in pairs] if transposed else pairs


def match_boxes(iou: list[list[float]], threshold: float) -> list[tuple[int, int]]:
    """Maximize the number of IoU >= threshold matches, then their total IoU."""
    if not iou or not iou[0]:
        return []
    # One extra valid match outweighs the largest possible sum of IoUs.
    bonus = min(len(iou), len(iou[0])) + 1
    weights = [[bonus + value if value >= threshold else 0.0 for value in row] for row in iou]
    return [(p, g) for p, g in hungarian_maximize(weights) if iou[p][g] >= threshold]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    center_distance_sum: float = 0.0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.iou_sum += other.iou_sum
        self.center_distance_sum += other.center_distance_sum

    def classification_report(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }

    def report(self) -> dict:
        return {
            "mean_iou": self.iou_sum / self.tp if self.tp else None,
            "mean_center_distance_px": self.center_distance_sum / self.tp if self.tp else None,
            **self.classification_report(),
        }


def evaluate_boxes(
    predictions: list[Box], ground_truth: list[Box], iou: list[list[float]],
    matches: list[tuple[int, int]],
) -> Counts:
    counts = Counts(
        tp=len(matches),
        fp=len(predictions) - len(matches),
        fn=len(ground_truth) - len(matches),
    )
    for pred_index, gt_index in matches:
        pred, gt = predictions[pred_index], ground_truth[gt_index]
        counts.iou_sum += iou[pred_index][gt_index]
        counts.center_distance_sum += math.hypot(
            (pred[0] + pred[2] - gt[0] - gt[2]) / 2,
            (pred[1] + pred[3] - gt[1] - gt[3]) / 2,
        )
    return counts


def evaluate_stacking(
    predicted_edges: set[tuple[str, str]],
    ground_truth_edges: set[tuple[AnnotationId, AnnotationId]],
    id_map: dict[str, AnnotationId],
) -> Counts:
    """Score directed edges after mapping graph IDs through matched boxes."""
    mapped_edges = {
        (id_map[upper], id_map[lower])
        for upper, lower in predicted_edges
        if upper in id_map and lower in id_map
    }
    tp = len(mapped_edges & ground_truth_edges)
    return Counts(tp=tp, fp=len(predicted_edges) - tp, fn=len(ground_truth_edges) - tp)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def evaluate_dataset(ground_truth_path: Path, stacking_path: Path, predictions_dir: Path) -> dict:
    ground_truth = load_ground_truth(ground_truth_path)
    stacking_ground_truth = load_stacking_ground_truth(stacking_path)
    prediction_files = sorted(predictions_dir.glob("*.debug.json"))
    if not prediction_files:
        raise ValueError(f"No *.debug.json prediction files found in {predictions_dir}")

    totals = {threshold: Counts() for threshold in THRESHOLDS}
    stacking_totals = {threshold: Counts() for threshold in THRESHOLDS}
    overall = {"ground_truth_boxes": 0, "prediction_boxes": 0, "missing_boxes": 0}
    scenes = {}
    evaluated_ground_truth = set()
    for path in prediction_files:
        scene = path.name.removesuffix(".debug.json")
        # The released held-out folders use held_01, while the bounding-box
        # annotations name the same image scene_heldout_01. Prefer exact names.
        gt_scene = scene
        if gt_scene not in ground_truth and scene.startswith("held_"):
            gt_scene = "scene_heldout_" + scene.removeprefix("held_")
        if gt_scene not in ground_truth:
            raise ValueError(f"{path}: no ground-truth image for scene {scene}")
        if gt_scene in evaluated_ground_truth:
            raise ValueError(f"{path}: ground-truth scene {gt_scene} already evaluated")
        evaluated_ground_truth.add(gt_scene)
        if gt_scene not in stacking_ground_truth:
            raise ValueError(f"{stacking_path}: no stacking annotations for scene {gt_scene}")
        annotations = ground_truth[gt_scene]
        annotation_ids = [annotation_id for annotation_id, _ in annotations]
        gt_boxes = [box for _, box in annotations]
        gt_edges = stacking_ground_truth[gt_scene]
        unknown_ids = {object_id for edge in gt_edges for object_id in edge} - set(annotation_ids)
        if unknown_ids:
            raise ValueError(f"{stacking_path}: {gt_scene} references unknown annotation IDs {unknown_ids}")
        predictions, missing = load_predictions(path, scene)
        predicted_ids = [object_id for object_id, _ in predictions]
        pred_boxes = [box for _, box in predictions]
        graph_path = path.with_name(f"{scene}.json")
        graph_ids, predicted_edges = load_predicted_stacking(graph_path)
        matrix = pairwise_iou(pred_boxes, gt_boxes)
        metrics = {}
        stacking_metrics = {}
        for threshold in THRESHOLDS:
            matches = match_boxes(matrix, threshold)
            counts = evaluate_boxes(pred_boxes, gt_boxes, matrix, matches)
            totals[threshold].add(counts)
            metrics[f"{threshold:.2f}"] = counts.report()
            id_map = {
                predicted_ids[pred_index]: annotation_ids[gt_index]
                for pred_index, gt_index in matches
                if predicted_ids[pred_index] in graph_ids
            }
            stacking_counts = evaluate_stacking(predicted_edges, gt_edges, id_map)
            stacking_totals[threshold].add(stacking_counts)
            stacking_metrics[f"{threshold:.2f}"] = {
                **stacking_counts.classification_report(),
                "object_id_mapping": id_map,
            }
        scenes[scene] = {
            "ground_truth_scene": gt_scene,
            "ground_truth_boxes": len(gt_boxes),
            "prediction_boxes": len(pred_boxes),
            "missing_boxes": missing,
            "thresholds": metrics,
            "stacking": {"thresholds": stacking_metrics},
        }
        overall["ground_truth_boxes"] += len(gt_boxes)
        overall["prediction_boxes"] += len(pred_boxes)
        overall["missing_boxes"] += missing

    overall["thresholds"] = {f"{t:.2f}": totals[t].report() for t in THRESHOLDS}
    overall["stacking"] = {
        "thresholds": {f"{t:.2f}": stacking_totals[t].classification_report() for t in THRESHOLDS}
    }
    return {
        "ground_truth": display_path(ground_truth_path),
        "stacking_ground_truth": display_path(stacking_path),
        "predictions_directory": display_path(predictions_dir),
        "evaluated_scenes": list(scenes),
        "excluded_ground_truth_scenes": sorted(set(ground_truth) - evaluated_ground_truth),
        "thresholds": list(THRESHOLDS),
        "scenes": scenes,
        "overall": overall,
    }


def print_table(report: dict) -> None:
    scene_width = max(12, *(len(scene) for scene in report["scenes"]))
    header = (
        f"{'Scene':<{scene_width}} {'Thr':>4} {'IoU':>7} {'Center px':>10} "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4}"
    )
    print(header)
    print("-" * len(header))
    for scene, result in [*report["scenes"].items(), ("ALL", report["overall"])]:
        for threshold, metrics in result["thresholds"].items():
            mean_iou = metrics["mean_iou"]
            distance = metrics["mean_center_distance_px"]
            iou_text = f"{mean_iou:.3f}" if mean_iou is not None else "-"
            distance_text = f"{distance:.1f}" if distance is not None else "-"
            print(
                f"{scene:<{scene_width}} {threshold:>4} {iou_text:>7} {distance_text:>10} "
                f"{metrics['precision']:>7.3f} {metrics['recall']:>7.3f} {metrics['f1']:>7.3f} "
                f"{metrics['tp']:>4} {metrics['fp']:>4} {metrics['fn']:>4}"
            )

    print("\nStacking relationships:")
    header = (
        f"{'Scene':<{scene_width}} {'Thr':>4} {'Prec':>7} {'Recall':>7} "
        f"{'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4}"
    )
    print(header)
    print("-" * len(header))
    for scene, result in [*report["scenes"].items(), ("ALL", report["overall"])]:
        for threshold, metrics in result["stacking"]["thresholds"].items():
            print(
                f"{scene:<{scene_width}} {threshold:>4} {metrics['precision']:>7.3f} "
                f"{metrics['recall']:>7.3f} {metrics['f1']:>7.3f} "
                f"{metrics['tp']:>4} {metrics['fp']:>4} {metrics['fn']:>4}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=PROJECT_ROOT / "dataset/ground_truth/bounding_boxes/ground_truth_bboxes.json",
        help="COCO ground-truth annotations",
    )
    parser.add_argument(
        "--stacking-ground-truth", type=Path,
        default=PROJECT_ROOT / "dataset/ground_truth/ground_truth_stacking.json",
        help="ground-truth stacking relationships by scene",
    )
    parser.add_argument(
        "--predictions", type=Path, default=PROJECT_ROOT / "results_dev_set/scene_graphs",
        help="directory containing *.debug.json predictions and corresponding scene graphs",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results_dev_set/perception_evaluation.json",
        help="path for the JSON report",
    )
    args = parser.parse_args()
    try:
        report = evaluate_dataset(args.ground_truth, args.stacking_ground_truth, args.predictions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print_table(report)
    print(
        f"\nEvaluated {len(report['evaluated_scenes'])} scenes; "
        f"excluded {len(report['excluded_ground_truth_scenes'])} without predictions. "
        f"Results written to {args.output}"
    )


if __name__ == "__main__":
    main()
