"""Check the grounded graph boundary and its submission and overlay consumers."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.results import plan_to_result
from src.scene_graph import derive_support_relations, supporting_ids
from src.schemas import load_object_extraction_schema, load_scene_discovery_schema
from src.validation import validate_scene_graph, validate_structure
from src.visualization import write_scene_overlay


def object_record(oid, label, kind, box, point):
    return {"id": oid, "label": label, "kind": kind, "color": None,
            "bbox_2d": box, "pick_point_2d": point, "confidence": 0.9,
            "relations": {"inside": None, "on_top_of": [], "supports": [],
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
        self.graph["objects"][0]["relations"]["on_top_of"] = ["tray_1"]
        supported = derive_support_relations(self.graph)
        self.assertEqual(supported["objects"][1]["relations"]["supports"], ["gear_1"])
        self.assertEqual(self.graph["objects"][1]["relations"]["supports"], [])
        self.assertIs(validate_scene_graph(supported), supported)

    def test_multiple_supports_have_independent_inverses_and_preserve_containment(self):
        self.graph["objects"].extend([
            object_record("gear_2", "yellow gear", "movable", [80, 30, 130, 90], [110, 60]),
            object_record("cube_1", "red cube", "movable", [40, 40, 110, 70], [75, 55]),
        ])
        self.graph["objects"][-1]["relations"].update(
            inside="tray_1", on_top_of=["gear_1", "gear_2"])
        graph = derive_support_relations(self.graph)
        self.assertEqual(graph["objects"][-1]["relations"]["on_top_of"], ["gear_1", "gear_2"])
        self.assertEqual(graph["objects"][0]["relations"]["supports"], ["cube_1"])
        self.assertEqual(graph["objects"][2]["relations"]["supports"], ["cube_1"])
        self.assertEqual(graph["objects"][1]["relations"]["supports"], [])
        self.assertIs(validate_scene_graph(graph), graph)

    def test_saved_single_parent_graphs_are_accepted_and_normalized_without_mutation(self):
        self.graph["objects"][0]["relations"]["on_top_of"] = "tray_1"
        self.graph["objects"][1]["relations"].update(on_top_of=None, supports=["gear_1"])
        self.assertIs(validate_scene_graph(self.graph), self.graph)
        graph = derive_support_relations(self.graph)
        self.assertEqual(graph["objects"][0]["relations"]["on_top_of"], ["tray_1"])
        self.assertEqual(graph["objects"][1]["relations"]["on_top_of"], [])
        self.assertEqual(self.graph["objects"][0]["relations"]["on_top_of"], "tray_1")
        self.assertIsNone(self.graph["objects"][1]["relations"]["on_top_of"])

    def test_multiple_support_references_duplicates_and_cycles_are_rejected(self):
        self.graph["objects"].append(
            object_record("gear_2", "yellow gear", "movable", [80, 30, 130, 90], [110, 60]))
        for supports in (["tray_1", "missing"], ["tray_1", "gear_1"], ["tray_1", "tray_1"]):
            invalid = deepcopy(self.graph)
            invalid["objects"][0]["relations"]["on_top_of"] = supports
            with self.subTest(supports=supports), self.assertRaises(ValueError):
                validate_scene_graph(invalid)
        self.graph["objects"][0]["relations"]["on_top_of"] = ["tray_1", "gear_2"]
        self.graph["objects"][2]["relations"]["on_top_of"] = ["gear_1"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_scene_graph(derive_support_relations(self.graph))

    def test_containment_and_support_union_cannot_form_a_cycle(self):
        self.graph["objects"][0]["relations"]["inside"] = "tray_1"
        self.graph["objects"][1]["relations"]["on_top_of"] = ["gear_1"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_scene_graph(derive_support_relations(self.graph))

    def test_supporting_ids_reads_current_and_saved_values(self):
        values = ["gear_1", "gear_2"]
        self.assertEqual(supporting_ids(values), values)
        self.assertIsNot(supporting_ids(values), values)
        self.assertEqual(supporting_ids("gear_1"), ["gear_1"])
        self.assertEqual(supporting_ids(None), [])
        for invalid in (1, True, {}, [None], ["gear_1", 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                supporting_ids(invalid)

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

    def test_unknown_confidence_is_valid_but_numeric_bounds_remain_enforced(self):
        self.graph["objects"][0]["confidence"] = None
        self.assertIs(validate_scene_graph(self.graph), self.graph)
        for confidence in (-0.01, 1.01, "unknown", True):
            with self.subTest(confidence=confidence):
                invalid = deepcopy(self.graph)
                invalid["objects"][0]["confidence"] = confidence
                with self.assertRaises(ValueError):
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
        invalid["objects"][0]["relations"]["on_top_of"] = ["tray_1"]
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
