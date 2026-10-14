"""Check the grounded graph boundary and its submission and overlay consumers."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.results import plan_to_result
from src.scene_graph import derive_support_relations
from src.schemas import load_object_extraction_schema, load_scene_discovery_schema
from src.validation import validate_scene_graph, validate_structure
from src.visualization import write_scene_overlay


def object_record(oid, label, kind, box, point):
    return {"id": oid, "label": label, "kind": kind, "color": None,
            "bbox_2d": box, "pick_point_2d": point, "confidence": 0.9,
            "relations": {"inside": None, "on_top_of": None, "supports": [],
                          "held_by_human": False, "partially_occluded": False}}


class SceneGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = {"image": {"width": 160, "height": 120}, "objects": [
            object_record("gear_1", "red gear", "movable", [20, 30, 80, 90], [50, 60]),
            object_record("tray_1", "grey tray", "container", [0, 0, 159, 119], [100, 100]),
        ]}

    def test_support_is_derived_without_mutation_or_containment_inference(self):
        self.graph["objects"][0]["relations"]["inside"] = "tray_1"
        original = deepcopy(self.graph)
        contained = derive_support_relations(self.graph)
        self.assertEqual(contained["objects"][1]["relations"]["supports"], [])
        self.assertEqual(self.graph, original)
        self.graph["objects"][0]["relations"]["on_top_of"] = "tray_1"
        supported = derive_support_relations(self.graph)
        self.assertEqual(supported["objects"][1]["relations"]["supports"], ["gear_1"])
        self.assertEqual(self.graph["objects"][1]["relations"]["supports"], [])
        self.assertIs(validate_scene_graph(supported), supported)

    def test_border_coordinates_are_inclusive_and_pick_stays_in_box(self):
        self.graph["objects"][1]["pick_point_2d"] = [159, 119]
        self.assertIs(validate_scene_graph(self.graph, 160, 120), self.graph)
        for box, point in (([0, 0, 160, 119], [159, 119]),
                           ([20, 30, 80, 90], [81, 60])):
            invalid = deepcopy(self.graph)
            invalid["objects"][0].update(bbox_2d=box, pick_point_2d=point)
            with self.subTest(box=box, point=point), self.assertRaises(ValueError):
                validate_scene_graph(invalid)

    def test_old_coordinate_format_and_non_json_numbers_are_rejected(self):
        invalid = deepcopy(self.graph)
        obj = invalid["objects"][0]
        obj["target_pixel"] = obj.pop("pick_point_2d")
        with self.assertRaises(ValueError):
            validate_scene_graph(invalid)
        invalid = deepcopy(self.graph)
        invalid["objects"][0]["confidence"] = float("nan")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            validate_scene_graph(invalid)

    def test_references_ids_and_inverse_supports_are_checked(self):
        invalid = deepcopy(self.graph)
        invalid["objects"].append(deepcopy(invalid["objects"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            validate_scene_graph(invalid)
        invalid = deepcopy(self.graph)
        invalid["objects"][0]["relations"]["inside"] = "missing_tray"
        with self.assertRaisesRegex(ValueError, "unknown object"):
            validate_scene_graph(invalid)
        invalid = deepcopy(self.graph)
        invalid["objects"][0]["relations"]["on_top_of"] = "tray_1"
        with self.assertRaisesRegex(ValueError, "supports"):
            validate_scene_graph(invalid)

    def test_submission_target_comes_from_first_pick_not_destination(self):
        plan = {"decision": "plan", "steps": [
            {"action": "pick", "object_id": "gear_1"},
            {"action": "place", "object_id": "tray_1"},
        ], "question": None, "reason": None}
        result = plan_to_result("scene_i1", plan, self.graph)
        self.assertEqual(result["target_pixel"], [50, 60])
        self.assertEqual(result["actions"], [
            {"action": "pick", "object": "gear_1"},
            {"action": "place", "object": "tray_1"},
        ])
        self.assertNotIn("target_pixel", self.graph["objects"][0])

    def test_overlay_draws_inclusive_box_and_pick_without_changing_rgb(self):
        rgb = np.full((120, 160, 3), [10, 20, 30], dtype=np.uint8)
        original = rgb.copy()
        graph = deepcopy(self.graph)
        graph["objects"] = graph["objects"][:1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlays" / "scene.png"
            write_scene_overlay(rgb, graph, path)
            overlay = cv2.imread(str(path))
        np.testing.assert_array_equal(rgb, original)
        self.assertEqual(overlay.shape, rgb.shape)
        background = [30, 20, 10]
        np.testing.assert_array_equal(overlay[110, 150], background)
        np.testing.assert_array_equal(overlay[60, 81], background)
        for u in (20, 80, 50):
            self.assertFalse(np.array_equal(overlay[60, u], background))


class ModelOutputContractTests(unittest.TestCase):
    def test_extraction_can_have_no_hints_but_discovery_requires_visible_labels(self):
        empty = {"objects": []}
        self.assertIs(validate_structure(empty, load_object_extraction_schema(), "Extraction"), empty)
        with self.assertRaises(ValueError):
            validate_structure(empty, load_scene_discovery_schema(), "Discovery")
        visible = {"objects": ["red gear", "grey tray"]}
        self.assertIs(validate_structure(visible, load_scene_discovery_schema(), "Discovery"), visible)

    def test_qwen_discovery_returns_labels_without_graph_coordinates(self):
        for invalid in ({"objects": [{"label": "gear", "bbox_2d": [0, 0, 20, 20]}]},
                        {"objects": ["gear"], "relations": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_structure(invalid, load_scene_discovery_schema(), "Discovery")


if __name__ == "__main__":
    unittest.main()
