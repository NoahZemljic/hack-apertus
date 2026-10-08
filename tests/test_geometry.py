"""Mask-based spatial reasoning contracts and conservative failure cases."""

from contextlib import chdir
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.geometry import GeometryConfig, GeometryReasoner, main


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.shape = (128, 256)
        self.config = GeometryConfig(reference_image_side=256)
        self.reasoner = GeometryReasoner(self.config, base_dir=self.root)
        cv2.imwrite(str(self.root / "rgb.png"), np.full((*self.shape, 3), 100, np.uint8))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def rectangle(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        mask = np.zeros(self.shape, np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    def scene(self, *masks: np.ndarray) -> dict:
        objects = []
        for i, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            oid = f"object_{i+1}"
            path = f"{oid}.png"
            cv2.imwrite(str(self.root / path), mask * 255)
            objects.append({"id": oid, "label": f"unchanged label {i}", "aliases": [f"valid alias {i}"],
                            "score": 0.85, "dino_score": 0.85, "sam_score": 0.9,
                            "box": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                            "mask_box": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                            "centroid": [float(xs.mean()), float(ys.mean())],
                            "safe_pixel": [int(xs[0]), int(ys[0])], "mask_area": int(xs.size),
                            "mask_path": path, "needs_review": False, "source_ids": [f"source_{i}"]})
        return {"scene": "arbitrary_scene", "image_path": "rgb.png", "objects": objects, "queries": ["kept"]}

    @staticmethod
    def edges(result: dict, key: str = "relations") -> set[tuple[str, str, str]]:
        return {(edge["source"], edge["type"], edge["target"]) for edge in result[key]}

    def test_preserves_all_original_metadata_without_mutation(self) -> None:
        scene = self.scene(self.rectangle(10, 10, 30, 30))
        original = deepcopy(scene)
        result, debug = self.reasoner.analyze_scene(scene)
        self.assertEqual(scene, original)
        for key, value in original["objects"][0].items():
            self.assertEqual(result["objects"][0][key], value)
        self.assertEqual(result["queries"], ["kept"])
        self.assertEqual(debug["pairs"], [])
        self.assertEqual(result["relation_candidates"], [])

    def test_left_right_are_local_neighbors_with_consistent_inverses(self) -> None:
        reasoner = GeometryReasoner(GeometryConfig(reference_image_side=256, direction_max_normalized_distance=10), base_dir=self.root)
        scene = self.scene(*(self.rectangle(x, 30, x+20, 60) for x in (10, 50, 90)))
        result, debug = reasoner.analyze_scene(scene)
        edges = self.edges(result)
        self.assertIn(("object_1", "left_of", "object_2"), edges)
        self.assertIn(("object_2", "right_of", "object_1"), edges)
        self.assertIn(("object_2", "left_of", "object_3"), edges)
        self.assertNotIn(("object_1", "left_of", "object_3"), edges)
        self.assertEqual(len(debug["pairs"]), 3)  # All measurements remain available.

    def test_relative_size_dead_zone_avoids_effectively_equal_centroids(self) -> None:
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(20, 20, 100, 80), self.rectangle(30, 25, 110, 85)))
        self.assertEqual(debug["pairs"][0]["centroid_dx"], 10)
        self.assertGreater(debug["pairs"][0]["horizontal_dead_zone"], 10)
        self.assertFalse(any(e[1] in ("left_of", "right_of", "above", "below") for e in self.edges(result)))

    def test_vertical_geometry_and_off_axis_pairs(self) -> None:
        result, _ = self.reasoner.analyze_scene(self.scene(self.rectangle(10, 10, 30, 30), self.rectangle(10, 50, 30, 70), self.rectangle(100, 90, 120, 110)))
        self.assertIn(("object_1", "above", "object_2"), self.edges(result))
        self.assertIn(("object_2", "below", "object_1"), self.edges(result))
        self.assertFalse(any(e[0] == "object_3" or e[2] == "object_3" for e in self.edges(result)))

    def test_overlap_measures_actual_intersection_and_iou(self) -> None:
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(10, 10, 30, 30), self.rectangle(20, 10, 40, 30)))
        pair = debug["pairs"][0]
        self.assertEqual(pair["intersection_pixels"], 200)
        self.assertEqual(pair["fraction_a_in_b"], 0.5)
        self.assertAlmostEqual(pair["mask_iou"], 1/3)
        self.assertIn(("object_1", "overlaps", "object_2"), self.edges(result))
        self.assertIn("object_1", result["objects"][1]["geometric_relations"]["overlaps"])
        self.assertEqual(result["relation_candidates"], [])

    def test_box_intersection_never_suffices_for_overlap_or_support(self) -> None:
        a, b = np.zeros(self.shape, np.uint8), np.zeros(self.shape, np.uint8)
        cv2.fillPoly(a, [np.array([[20, 20], [80, 20], [20, 80]])], 1)
        cv2.fillPoly(b, [np.array([[80, 80], [80, 40], [40, 80]])], 1)
        result, debug = self.reasoner.analyze_scene(self.scene(a, b))
        self.assertEqual(debug["pairs"][0]["intersection_pixels"], 0)
        self.assertFalse(any(e[1] == "overlaps" for e in self.edges(result)))
        self.assertEqual(result["relation_candidates"], [])

    def test_containment_is_image_geometry_and_does_not_establish_support(self) -> None:
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(20, 10, 120, 110), self.rectangle(50, 40, 70, 60)))
        edges = self.edges(result)
        self.assertIn(("object_2", "inside", "object_1"), edges)
        self.assertIn(("object_1", "contains", "object_2"), edges)
        self.assertFalse(any(e[1] in ("left_of", "right_of", "above", "below", "touching") for e in edges))
        self.assertEqual(result["relation_candidates"], [])
        self.assertEqual(debug["pairs"][0]["fraction_b_in_a"], 1)
        self.assertTrue(all(e["scope"] == "image_plane" for e in result["relations"]))

    def test_enclosure_recovers_occlusion_holes_and_records_actual_zero_overlap(self) -> None:
        ring = self.rectangle(10, 10, 110, 110)
        ring[20:100, 20:100] = 0
        result, debug = self.reasoner.analyze_scene(self.scene(ring, self.rectangle(40, 40, 60, 60)))
        pair = debug["pairs"][0]
        self.assertEqual(pair["fraction_b_in_a"], 0)
        self.assertEqual(pair["fraction_b_in_enclosed_a"], 1)
        inside = next(e for e in result["relations"] if e["type"] == "inside")
        self.assertEqual(inside["evidence"], "enclosed_mask_region")
        self.assertEqual(result["relation_candidates"], [])

    def test_open_shape_and_convex_hull_do_not_imply_containment(self) -> None:
        u_shape = self.rectangle(10, 10, 110, 110)
        u_shape[10:100, 20:100] = 0
        result, debug = self.reasoner.analyze_scene(self.scene(u_shape, self.rectangle(40, 40, 60, 60)))
        self.assertEqual(debug["pairs"][0]["fraction_b_in_hull_a"], 1)
        self.assertEqual(debug["pairs"][0]["fraction_b_in_enclosed_a"], 0)
        self.assertFalse(any(e[1] in ("inside", "contains") for e in self.edges(result)))

    def test_equal_masks_do_not_create_mutual_containment_or_support(self) -> None:
        mask = self.rectangle(30, 30, 70, 70)
        result, _ = self.reasoner.analyze_scene(self.scene(mask, mask))
        self.assertEqual(self.edges(result), {("object_1", "overlaps", "object_2")})
        self.assertEqual(result["relation_candidates"], [])

    def test_touching_uses_outer_boundary_tolerance_with_actual_distance(self) -> None:
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(20, 30, 40, 60), self.rectangle(42, 30, 62, 60)))
        self.assertIn(("object_1", "touching", "object_2"), self.edges(result))
        self.assertEqual(debug["pairs"][0]["outer_boundary_a_to_b"]["min_distance"], 3)
        result, _ = self.reasoner.analyze_scene(self.scene(self.rectangle(20, 30, 40, 60), self.rectangle(44, 30, 64, 60)))
        self.assertFalse(any(e[1] == "touching" for e in self.edges(result)))

    def test_contact_distance_scales_but_has_one_pixel_floor(self) -> None:
        reasoner = GeometryReasoner(base_dir=self.root)
        result, debug = reasoner.analyze_scene(self.scene(self.rectangle(20, 30, 40, 60), self.rectangle(40, 30, 60, 60)))
        self.assertEqual(debug["effective_touch_tolerance"], 1)
        self.assertIn(("object_1", "touching", "object_2"), self.edges(result))

    def test_side_stack_requires_opposing_real_mask_edges(self) -> None:
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(30, 20, 70, 40), self.rectangle(30, 40, 70, 80)))
        self.assertIn(("object_1", "on", "object_2"), self.edges(result, "relation_candidates"))
        self.assertIn(("object_2", "under", "object_1"), self.edges(result, "relation_candidates"))
        self.assertEqual(debug["pairs"][0]["support_profile_a_on_b"]["contact_fraction"], 1)
        self.assertTrue(all(e["needs_review"] for e in result["relation_candidates"]))
        self.assertFalse(any("on_top_of" in obj for obj in result["objects"]))
        # Same aligned boxes with a substantial gap do not establish support.
        result, _ = self.reasoner.analyze_scene(self.scene(self.rectangle(30, 10, 70, 30), self.rectangle(30, 40, 70, 80)))
        self.assertEqual(result["relation_candidates"], [])

    def test_offset_stack_requires_actual_overlap_boundary_and_compactness(self) -> None:
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(80, 20, 160, 100), self.rectangle(103, 65, 138, 100)))
        self.assertIn(("object_2", "on", "object_1"), self.edges(result, "relation_candidates"))
        on = next(e for e in result["relation_candidates"] if e["type"] == "on")
        self.assertEqual(on["evidence"], "offset_overlap")
        self.assertEqual(on["scope"], "physical_hypothesis")
        self.assertGreater(debug["pairs"][0]["outer_boundary_b_to_a"]["contact_fraction"], 0.15)
        # A centered contained object has no directional boundary evidence.
        result, _ = self.reasoner.analyze_scene(self.scene(self.rectangle(80, 20, 160, 100), self.rectangle(103, 43, 138, 78)))
        self.assertEqual(result["relation_candidates"], [])
        # An elongated enclosing mask is not a compact stack support.
        result, _ = self.reasoner.analyze_scene(self.scene(self.rectangle(20, 30, 220, 100), self.rectangle(90, 55, 140, 100)))
        self.assertEqual(result["relation_candidates"], [])

    def test_similarly_supported_candidates_cause_abstention(self) -> None:
        lower = self.rectangle(30, 40, 70, 80)
        result, debug = self.reasoner.analyze_scene(self.scene(self.rectangle(30, 20, 70, 40), lower, lower))
        self.assertEqual(result["relation_candidates"], [])
        proposals = [proposal for pair in debug["pairs"] for proposal in pair["stacking"] if proposal["source"] == "object_1"]
        self.assertTrue(all(p["selection"] == "ambiguous" for p in proposals))

    def test_input_review_flags_are_preserved_on_relations(self) -> None:
        scene = self.scene(self.rectangle(10, 10, 30, 30), self.rectangle(40, 10, 60, 30))
        scene["objects"][0]["needs_review"] = True
        scene["objects"][0]["review_reasons"] = ["existing_ambiguous_label"]
        result, _ = self.reasoner.analyze_scene(scene)
        self.assertTrue(all(edge["needs_review"] and edge["input_needs_review"] for edge in result["relations"]))
        self.assertEqual(result["objects"][0]["review_reasons"], ["existing_ambiguous_label"])

    def test_stale_or_invalid_segmentation_is_rejected(self) -> None:
        for key, value, message in (("mask_area", 1, "mask_area"), ("centroid", [0, 0], "centroid"),
                                    ("safe_pixel", [0, 0], "safe_pixel"), ("mask_box", [0, 0, 2, 2], "mask_box")):
            with self.subTest(key=key):
                scene = self.scene(self.rectangle(10, 10, 30, 30))
                scene["objects"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.reasoner.analyze_scene(scene)

    def test_missing_empty_wrong_size_and_nonbinary_masks_fail(self) -> None:
        scene = self.scene(self.rectangle(10, 10, 30, 30))
        path = self.root / scene["objects"][0]["mask_path"]
        for raw, message in ((np.zeros(self.shape, np.uint8), "empty"),
                              (np.ones((10, 10), np.uint8), "binary mask"),
                              (np.full(self.shape, 128, np.uint8), "binary mask")):
            with self.subTest(message=message):
                cv2.imwrite(str(path), raw)
                with self.assertRaisesRegex(ValueError, message):
                    self.reasoner.analyze_scene(scene)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.reasoner.analyze_scene(scene)

    def test_cli_outputs_measurement_links_and_visualizations_with_relative_paths(self) -> None:
        scene = self.scene(self.rectangle(30, 20, 70, 40), self.rectangle(30, 40, 70, 80))
        path = self.root / "segmentation_results.json"
        path.write_text(json.dumps([scene]))
        config = self.root / "thresholds.json"
        config.write_text(json.dumps({"reference_image_side": 256}))
        with chdir(self.root):
            self.assertEqual(main(["--config", str(config)]), 0)
        result = json.loads((self.root / "geometry_results.json").read_text())[0]
        debug = json.loads((self.root / "geometry_debug.json").read_text())
        self.assertEqual(debug["config"]["reference_image_side"], 256)
        for edge in result["relations"] + result["relation_candidates"]:
            pair = debug["scenes"][0]["pairs"][edge["measurements_index"]]
            self.assertEqual({pair["a"], pair["b"]}, {edge["source"], edge["target"]})
        for suffix in ("", "_directions"):
            image_path = self.root / "outputs/geometry_debug" / f"arbitrary_scene{suffix}.png"
            self.assertIsNotNone(cv2.imread(str(image_path)))
        self.assertNotIn("mask", result["objects"][0])

    def test_config_and_duplicate_ids_are_validated(self) -> None:
        for values in ({"inside_min_area_ratio": 1}, {"touch_min_boundary_fraction": -1},
                       {"direction_neighbors": 0}, {"reference_image_side": 0}, {"overlap_min_fraction": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GeometryConfig(**values)
        scene = self.scene(self.rectangle(10, 10, 30, 30))
        scene["objects"].append(deepcopy(scene["objects"][0]))
        path = self.root / "input.json"
        path.write_text(json.dumps([scene]))
        with self.assertRaisesRegex(ValueError, "duplicate object ID"):
            self.reasoner.load_detections(path)

    def test_centroid_tolerance_does_not_allow_coordinates_outside_image(self) -> None:
        scene = self.scene(self.rectangle(0, 20, 1, 40))
        scene["objects"][0]["centroid"][0] = -0.001
        with self.assertRaisesRegex(ValueError, "centroid"):
            self.reasoner.analyze_scene(scene)


if __name__ == "__main__":
    unittest.main()
