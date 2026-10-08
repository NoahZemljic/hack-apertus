"""SAM 2 geometry, metadata, validation, and output-contract regressions."""

import copy
import json
import tempfile
import unittest
import warnings
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from src.masking.sam import DEFAULT_DETECTIONS, SamSegmenter, clip_box, main


class SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.segmenter = SamSegmenter()
        self.detection = {
            "id": "object_1", "label": "valid object", "aliases": ["valid object"],
            "score": 0.856, "box": [0, 0, 20, 20],
            "needs_review": True, "review_reasons": ["ambiguous_compound_label"],
            "source_ids": ["original_1"],
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def mask() -> np.ndarray:
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True
        return mask

    def test_geometry_matches_foreground_and_preserves_xy_order(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:7, 11:20] = True
        geometry = self.segmenter.calculate_geometry(mask)
        self.assertEqual(geometry["mask_area"], 45)
        self.assertEqual(geometry["mask_box"], [11, 2, 20, 7])
        self.assertEqual(geometry["centroid"], [15.0, 4.0])
        x, y = geometry["safe_pixel"]
        self.assertTrue(mask[y, x])
        self.assertEqual(geometry["safe_pixel_clearance"], 3.0)

    def test_safe_pixel_avoids_hole_and_maximizes_boundary_distance(self) -> None:
        mask = np.ones((21, 21), dtype=bool)
        mask[6:15, 6:15] = False
        geometry = self.segmenter.calculate_geometry(mask)
        self.assertEqual(geometry["centroid"], [10.0, 10.0])
        self.assertFalse(mask[10, 10])
        x, y = geometry["safe_pixel"]
        self.assertTrue(mask[y, x])
        distances = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2,
                                          cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        self.assertAlmostEqual(float(distances[y, x]), float(distances.max()))
        self.segmenter.validate_mask(mask, [0, 0, 21, 21], geometry)

    def test_image_edge_is_a_boundary_for_full_image_mask(self) -> None:
        geometry = self.segmenter.calculate_geometry(np.ones((9, 9), dtype=bool))
        self.assertEqual(geometry["safe_pixel"], [4, 4])
        self.assertEqual(geometry["safe_pixel_clearance"], 5.0)

    def test_disconnected_mask_safe_pixel_is_in_the_deepest_component(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[1:4, 1:4] = True
        mask[10:19, 20:29] = True
        geometry = self.segmenter.calculate_geometry(mask)
        self.assertEqual(geometry["mask_area"], 90)
        self.assertEqual(geometry["safe_pixel"], [24, 14])

    def test_best_nonempty_candidate_preserves_every_dino_field(self) -> None:
        masks = np.stack([self.mask(), np.zeros((20, 20), dtype=bool), np.ones((20, 20), dtype=bool)])
        original = copy.deepcopy(self.detection)
        result = self.segmenter.segment_object(self.detection, masks, np.array([0.8, 0.99, 0.7]))
        self.assertEqual(self.detection, original)
        for key, value in original.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["sam_score"], 0.8)
        self.assertEqual(result["dino_score"], original["score"])
        np.testing.assert_array_equal(result["mask"], masks[0])
        self.assertEqual(result["validation_warnings"], [])

    def test_empty_candidates_fail_with_id_instead_of_skipping_detection(self) -> None:
        with self.assertRaisesRegex(ValueError, "object_1.*no nonempty mask"):
            self.segmenter.segment_object(self.detection, np.zeros((3, 20, 20), dtype=bool),
                                          np.array([0.9, 0.8, 0.7]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.segmenter.calculate_geometry(np.zeros((20, 20), dtype=bool))

    def test_nonfinite_candidate_score_is_not_selected(self) -> None:
        result = self.segmenter.segment_object(self.detection, np.stack([self.mask(), self.mask()]),
                                              np.array([np.nan, 0.75]))
        self.assertEqual(result["sam_score"], 0.75)

    def test_invalid_safe_pixel_or_centroid_is_rejected(self) -> None:
        mask = self.mask()
        for key, value, message in [
            ("safe_pixel", [0, 0], "Safe pixel"),
            ("safe_pixel", [20, 0], "Safe pixel"),
            ("centroid", [20.0, 10.0], "centroid"),
            ("centroid", [float("nan"), 10.0], "centroid"),
        ]:
            with self.subTest(key=key, value=value):
                geometry = {**self.segmenter.calculate_geometry(mask), key: value}
                with self.assertRaisesRegex(ValueError, message):
                    self.segmenter.validate_mask(mask, self.detection["box"], geometry)

    def test_small_mask_warns_and_stays_present(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.segmenter.segment_object(self.detection, mask[None], np.array([0.8]))
        self.assertEqual(result["mask_area"], 1)
        self.assertEqual(result["safe_pixel"], [10, 10])
        self.assertIn("fills only", result["validation_warnings"][0])
        self.assertTrue(caught)

    def test_spill_is_reported_without_clipping_the_mask(self) -> None:
        mask = np.ones((20, 20), dtype=bool)
        detection = {**self.detection, "box": [7, 7, 13, 13]}
        with self.assertWarnsRegex(RuntimeWarning, "far outside"):
            result = self.segmenter.segment_object(detection, mask[None], np.array([0.9]))
        self.assertEqual(result["mask_area"], 400)
        self.assertEqual(result["mask_box"], [0, 0, 20, 20])
        self.assertEqual(result["box"], detection["box"])
        self.assertGreater(result["far_outside_box_fraction"], 0.1)

    def test_prompt_clipping_preserves_original_box_metadata(self) -> None:
        detection = {**self.detection, "box": [-1, 0, 21, 20]}
        self.assertEqual(clip_box(detection["box"], (20, 20)), [0, 0, 20, 20])
        with self.assertWarnsRegex(RuntimeWarning, "prompt was clipped"):
            result = self.segmenter.segment_object(detection, self.mask()[None], np.array([0.8]))
        self.assertEqual(result["box"], [-1, 0, 21, 20])

    def test_raw_metadata_or_duplicate_ids_are_rejected(self) -> None:
        path = self.root / "detections.json"
        for detections, message in [
            ([{key: value for key, value in self.detection.items() if key != "aliases"}], "cleaned"),
            ([self.detection, self.detection], "duplicate object ID"),
            ([{**self.detection, "id": "../escape"}], "filename component"),
        ]:
            with self.subTest(message=message):
                path.write_text(json.dumps([{"scene": "example", "detections": detections}]))
                with self.assertRaisesRegex(ValueError, message):
                    self.segmenter.load_detections(path)

    def test_mask_png_and_json_geometry_round_trip(self) -> None:
        result = self.segmenter.segment_object(self.detection, self.mask()[None], np.array([0.9]))
        mask_path = self.segmenter.save_mask(result["mask"], self.root / "masks" / "object_1.png")
        result["mask_path"] = mask_path.as_posix()
        output = self.root / "results.json"
        self.segmenter.save_results([{"scene": "example", "objects": [result]}], output)
        saved = json.loads(output.read_text())[0]["objects"][0]
        self.assertNotIn("mask", saved)
        for key in self.detection:
            self.assertEqual(saved[key], self.detection[key])
        with Image.open(saved["mask_path"]) as image:
            loaded = np.array(image)
            self.assertEqual(image.size, (20, 20))
        self.assertEqual(set(np.unique(loaded)), {0, 255})
        self.assertEqual(int((loaded > 0).sum()), saved["mask_area"])
        x, y = saved["safe_pixel"]
        self.assertEqual(int(loaded[y, x]), 255)

    def test_actual_sam2_processor_batches_boxes_and_restores_resolution(self) -> None:
        import torch
        from transformers import Sam2ImageProcessorFast, Sam2Processor

        image_path = self.root / "rgb.png"
        Image.new("RGB", (40, 30)).save(image_path)
        segmenter = SamSegmenter()
        segmenter.device = "cpu"
        segmenter.processor = Sam2Processor(Sam2ImageProcessorFast(size={"height": 32, "width": 32}))
        predictions = torch.full((1, 2, 3, 8, 8), -10.0)
        predictions[0, 0, 0, 1:4, 1:4] = 10.0
        predictions[0, 1, 2, 4:7, 4:7] = 10.0
        segmenter.model = Mock(return_value=SimpleNamespace(
            pred_masks=predictions, iou_scores=torch.tensor([[[0.8, 0.9, 0.7], [0.6, 0.7, 0.9]]]),
        ))
        detections = [self.detection, {**self.detection, "id": "object_2", "box": [20, 15, 40, 30]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = segmenter.segment_scene(image_path, detections)
        self.assertEqual([obj["id"] for obj in results], ["object_1", "object_2"])
        self.assertTrue(all(obj["mask"].shape == (30, 40) for obj in results))
        calls = segmenter.model.call_args.kwargs
        self.assertTrue(calls["multimask_output"])
        self.assertEqual(tuple(calls["input_boxes"].shape), (1, 2, 4))
        np.testing.assert_allclose(calls["input_boxes"][0, 1].numpy(), [16, 16, 32, 32])

    def test_default_cli_uses_cleaned_detections_and_required_output_layout(self) -> None:
        self.assertEqual(DEFAULT_DETECTIONS, Path("grounding_detections_clean.json"))
        scene_dir = self.root / "dataset" / "example"
        scene_dir.mkdir(parents=True)
        Image.new("RGB", (20, 20)).save(scene_dir / "rgb.png")
        scene = {"scene": "example", "queries": ["valid object"], "detections": [self.detection]}
        (self.root / DEFAULT_DETECTIONS).write_text(json.dumps([scene]))
        (self.root / "grounding_detections.json").write_text("invalid raw input must not be read")
        result = self.segmenter.segment_object(self.detection, self.mask()[None], np.array([0.9]))
        with chdir(self.root), patch("sys.argv", ["sam"]), \
                patch.object(SamSegmenter, "segment_scene", return_value=[result]) as segment:
            main()
        segment.assert_called_once_with(Path("dataset/example/rgb.png"), [self.detection])
        saved = json.loads((self.root / "segmentation_results.json").read_text())
        self.assertEqual(saved[0]["queries"], scene["queries"])
        self.assertEqual(saved[0]["objects"][0]["mask_path"], "outputs/segmentation/example/object_1.png")
        self.assertNotIn("mask", saved[0]["objects"][0])
        self.assertTrue((self.root / "outputs/segmentation/example/object_1.png").exists())
        with Image.open(self.root / "outputs/segmentation_debug/example.png") as image:
            self.assertEqual(image.format, "PNG")

    def test_empty_scene_does_not_load_model(self) -> None:
        image_path = self.root / "rgb.png"
        Image.new("RGB", (20, 20)).save(image_path)
        with patch.object(self.segmenter, "_load_model") as load:
            self.assertEqual(self.segmenter.segment_scene(image_path, []), [])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
