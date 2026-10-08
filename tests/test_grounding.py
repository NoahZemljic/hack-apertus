"""Regression checks for conservative DINO cleanup and its script integration."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.grounding.grounding_dino import GroundingDinoDetector, main
from src.grounding.postprocess_grounding import process_scene, query_candidates


class GroundingCleanupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image = self.root / "rgb.png"
        Image.new("RGB", (100, 100)).save(self.image)

    def tearDown(self):
        self.directory.cleanup()

    def process(self, queries, detections):
        item = {"scene": "example", "queries": queries, "detections": detections}
        original = copy.deepcopy(item)
        cleaned, report = process_scene(item, self.image)
        self.assertEqual(item, original)
        self.assertEqual(report["raw_count"] - report["clean_count"], report["dropped_count"])
        return cleaned["detections"], report

    @staticmethod
    def detection(label, box=None, score=0.8, object_id="original_1"):
        return {"label": label, "box": box or [10, 10, 30, 30], "score": score, "id": object_id}

    def test_valid_specific_and_unknown_names_are_preserved(self):
        for label, queries in [
            ("green mug", ["green mug", "mug"]),
            ("red gear", ["red gear", "gear"]),
            ("heart plate", ["plate", "heart plate", "coaster"]),
            ("Steel Plate", ["plate"]),
            ("unlisted object", ["mug"]),
        ]:
            with self.subTest(label=label):
                cleaned, _ = self.process(queries, [self.detection(label)])
                self.assertEqual(cleaned[0]["label"], label)
                self.assertFalse(cleaned[0]["needs_review"])

    def test_exact_compound_query_is_valid(self):
        self.assertEqual(
            query_candidates("red gear yellow gear", ["gear", "red gear", "yellow gear", "red gear yellow gear"]),
            ["red gear yellow gear"],
        )

    def test_generic_query_can_also_be_a_separate_compound_phrase(self):
        self.assertEqual(query_candidates("plate yellow plate", ["plate", "yellow plate"]),
                         ["plate", "yellow plate"])
        cleaned, _ = self.process(["plate", "yellow plate"], [self.detection("plate yellow plate")])
        self.assertEqual(cleaned[0]["label"], "plate yellow plate")
        self.assertTrue(cleaned[0]["needs_review"])

    def test_other_instances_do_not_resolve_or_delete_compounds(self):
        cleaned, _ = self.process(["red gear", "yellow gear", "gear"], [
            self.detection("red gear", [0, 0, 20, 20], object_id="red_1"),
            self.detection("yellow gear", [40, 40, 60, 60], object_id="yellow_1"),
            self.detection("red gear yellow gear", [70, 70, 90, 90], object_id="compound_1"),
        ])
        compound = next(d for d in cleaned if d["label"] == "red gear yellow gear")
        self.assertTrue(compound["needs_review"])
        self.assertEqual(compound["aliases"], ["red gear", "yellow gear"])
        self.assertEqual(len(cleaned), 3)

    def test_missing_query_does_not_force_a_new_name(self):
        cleaned, _ = self.process(["mandarin", "red cube"], [
            self.detection("red cube", [0, 0, 20, 20], object_id="red_cube_1"),
            self.detection("mandarin red cube", [50, 50, 70, 70]),
        ])
        self.assertEqual(cleaned[1]["label"], "mandarin red cube")
        self.assertTrue(cleaned[1]["needs_review"])

    def test_only_repeated_complete_queries_are_relabelled(self):
        cleaned, _ = self.process(["red gear", "gear"], [self.detection("red gear red gear")])
        self.assertEqual(cleaned[0]["label"], "red gear")
        self.assertEqual(cleaned[0]["raw_labels"], ["red gear red gear"])

    def test_duplicate_keeps_strongest_original_name_and_provenance(self):
        cleaned, report = self.process(["plate", "saucer", "heart plate"], [
            self.detection("plate", score=0.9, object_id="plate_1"),
            self.detection("saucer", score=0.7, object_id="saucer_1"),
            self.detection("heart plate", score=0.3, object_id="heart_plate_1"),
        ])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["label"], "plate")
        self.assertEqual(cleaned[0]["source_ids"], ["plate_1", "saucer_1", "heart_plate_1"])
        self.assertIn("conflicting_duplicate_labels", cleaned[0]["review_reasons"])
        self.assertEqual(report["merged_count"], 2)
        self.assertTrue(all(d["drop_reason"] == "duplicate_box" for d in report["dropped"]))

    def test_iou_chains_do_not_merge_distinct_endpoints(self):
        cleaned, _ = self.process(["mug"], [
            self.detection("mug", [0, 0, 20, 20], 0.9, "mug_1"),
            self.detection("mug", [1, 0, 21, 20], 0.8, "mug_2"),
            self.detection("mug", [2, 0, 22, 20], 0.7, "mug_3"),
        ])
        self.assertEqual(len(cleaned), 2)

    def test_clipping_preserves_exclusive_image_edge(self):
        cleaned, report = self.process(["tray"], [self.detection("tray", [-4, 0, 101, 100])])
        self.assertEqual(cleaned[0]["box"], [0, 0, 100, 100])
        self.assertEqual(report["clipped_count"], 1)

    def test_invalid_boxes_are_dropped(self):
        for box in [[20, 0, 10, 10], [-5, 0, -1, 10], [0, 0, 1], [0, 0, float("inf"), 10]]:
            with self.subTest(box=box):
                item = {"scene": "example", "queries": [], "detections": [self.detection("mug", box)]}
                cleaned, report = process_scene(item, self.image)
                self.assertEqual(cleaned["detections"], [])
                self.assertEqual(report["dropped"][0]["drop_reason"], "invalid_box")

    def test_area_filter_is_limited_to_small_object_failures(self):
        for label in ["yellow gear", "red cube", "electronic component"]:
            with self.subTest(label=label):
                cleaned, report = self.process([label], [self.detection(label, [0, 0, 100, 100])])
                self.assertEqual(cleaned, [])
                self.assertEqual(report["dropped"][0]["drop_reason"], "pathologically_large_box")
        for label in ["plate", "unknown object", "tray", "arm", "hand", "charm"]:
            with self.subTest(label=label):
                cleaned, _ = self.process([label], [self.detection(label, [0, 0, 100, 100])])
                self.assertEqual(cleaned[0]["label"], label)
                self.assertEqual(cleaned[0]["needs_review"], label in {"plate", "unknown object", "charm"})

    def test_empty_scene(self):
        cleaned, report = self.process([], [])
        self.assertEqual(cleaned, [])
        self.assertEqual(report["review_ids"], [])

    def test_script_runs_cleanup_immediately_after_detection(self):
        scene_dir = self.root / "example"
        scene_dir.mkdir()
        self.image.rename(scene_dir / "rgb.png")
        queries = self.root / "queries.json"
        queries.write_text(json.dumps([{"scene": "example", "qwen_objects": ["mug"]}]))
        raw = self.detection("mug", [-1, 0, 20, 20], object_id="mug_1")
        arguments = [
            "grounding_dino", "--queries", str(queries), "--dataset", str(self.root),
            "--output", str(self.root / "raw.json"),
            "--clean-output", str(self.root / "clean.json"),
            "--report", str(self.root / "report.json"),
            "--debug-dir", str(self.root / "plots"),
        ]
        with patch("sys.argv", arguments), patch.object(GroundingDinoDetector, "__init__", return_value=None), \
                patch.object(GroundingDinoDetector, "detect", return_value=[raw]) as detect:
            main()
        detect.assert_called_once()
        self.assertEqual(json.loads((self.root / "raw.json").read_text())[0]["detections"], [raw])
        cleaned = json.loads((self.root / "clean.json").read_text())[0]["detections"]
        self.assertEqual(cleaned[0]["box"], [0, 0, 20, 20])
        for suffix in ["boxes", "boxes_clean", "comparison"]:
            plot = self.root / "plots" / f"example_{suffix}.png"
            with Image.open(plot) as image:
                self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
