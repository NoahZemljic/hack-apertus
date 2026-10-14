"""Check the planning prompt against the two scene graph payloads in use."""

from copy import deepcopy
import json
import re
import unittest

from src.create_scene_graph import create_scene_graph
from src.prompting import derived_geometry, prepare_planning_prompt


class PlanningPromptTests(unittest.TestCase):
    def compact_graph(self):
        return create_scene_graph([
            {"id": "red_gear_1", "label": "red_gear", "target_pixel": [10, 20],
             "width_mm": 30, "graspable": True, "bbox_2d": [0, 0, 30, 40]},
            {"id": "grey_tray_1", "label": "grey_tray", "target_pixel": [40, 60],
             "width_mm": 200, "graspable": False},
        ], [{"subject": "red_gear_1", "relation": "inside", "object": "grey_tray_1", "confidence": 0.8}])

    def payload(self, prompt, heading):
        return json.loads(prompt.split(heading + "\n", 1)[1].split("\n\n", 1)[0])

    def assert_field_guidance_exists_in_graph(self, prompt, graph):
        for path in re.findall(r"^- (objects\[\]\.[a-z_0-9.]+):", prompt, re.MULTILINE):
            keys = path.split(".")[1:]
            values = graph["objects"]
            for key in keys:
                values = [value[key] for value in values if key in value]
            self.assertTrue(values, path)

    def test_current_pipeline_graph_is_preserved_and_only_supplied_fields_are_described(self):
        graph = self.compact_graph()
        before = deepcopy(graph)
        instruction = 'put the red gear in the grey tray, not "that mug"'
        prompt = prepare_planning_prompt(graph, instruction, 85)
        self.assertEqual(self.payload(prompt, "Scene graph:"), graph)
        self.assertEqual(graph, before)
        self.assertIn("Instruction: " + json.dumps(instruction), prompt)
        self.assert_field_guidance_exists_in_graph(prompt, graph)
        for absent in ("pick_point_2d", "held_by_human", "on_top_of", "partially_occluded", "objects[].kind", "objects[].color"):
            self.assertNotIn(absent, prompt)
        geometry = self.payload(prompt, "Deterministically derived 2D geometry:")
        self.assertEqual(geometry["left_to_right"], ["red_gear_1", "grey_tray_1"])
        self.assertEqual(geometry["pairwise_distances"], [{"a": "red_gear_1", "b": "grey_tray_1", "distance_px": 50.0}])

    def test_optional_measurements_are_not_described_when_missing(self):
        graph = self.compact_graph()
        for obj in graph["objects"]:
            del obj["width_mm"]
            del obj["graspable"]
        prompt = prepare_planning_prompt(graph, "pick the red gear", None)
        self.assertNotIn("width_mm", prompt)
        self.assertNotIn("graspable", prompt)
        self.assert_field_guidance_exists_in_graph(prompt, graph)

    def test_grounded_graph_preserves_supplied_centroid_geometry_and_relation_direction(self):
        graph = {"image": {"width": 100, "height": 80}, "objects": [
            {"id": "red_gear_1", "label": "red gear", "kind": "movable", "color": "red",
             "bbox_2d": [0, 0, 30, 40], "pick_point_2d": [10, 20], "confidence": 0.9,
             "relations": {"inside": None, "on_top_of": "yellow_gear_1", "supports": [],
                           "held_by_human": False, "partially_occluded": False}},
            {"id": "yellow_gear_1", "label": "yellow gear", "kind": "movable", "color": "yellow",
             "bbox_2d": [0, 0, 50, 50], "pick_point_2d": [40, 40], "confidence": 0.8,
             "relations": {"inside": None, "on_top_of": None, "supports": ["red_gear_1"],
                           "held_by_human": False, "partially_occluded": True}},
        ]}
        spatial = {"centroids_2d": {"red_gear_1": [25, 30], "yellow_gear_1": [20, 20]},
                   "left_to_right": ["yellow_gear_1", "red_gear_1"], "pairwise_distances": [],
                   "image_plane_relations": []}
        before = deepcopy((graph, spatial))
        prompt = prepare_planning_prompt(graph, "pick the bottom gear", 85, geometry=spatial)
        self.assertEqual(self.payload(prompt, "Scene graph:"), graph)
        self.assertEqual(self.payload(prompt, "Deterministically derived 2D geometry:"), spatial)
        self.assertEqual((graph, spatial), before)
        self.assert_field_guidance_exists_in_graph(prompt, graph)
        for absent in ("width_mm", "graspable", "target_pixel", "needs_review", "review_reasons"):
            self.assertNotIn(absent, prompt)
        self.assertNotIn("relations is a list", prompt)
        self.assertEqual(derived_geometry(graph)["left_to_right"], ["red_gear_1", "yellow_gear_1"])

    def test_empty_scene_does_not_require_coordinates_or_invent_objects(self):
        graph = {"objects": [], "relations": []}
        prompt = prepare_planning_prompt(graph, "pick the blue cube", 85)
        self.assertEqual(self.payload(prompt, "Scene graph:"), graph)
        self.assertEqual(self.payload(prompt, "Deterministically derived 2D geometry:")["pairwise_distances"], [])

    def test_distance_order_uses_centroids_instead_of_pick_points(self):
        graph = {"objects": [
            {"id": "anchor", "kind": "movable", "label": "mug", "pick_point_2d": [10, 0]},
            {"id": "near", "kind": "movable", "label": "yellow gear", "pick_point_2d": [100, 0]},
            {"id": "far", "kind": "movable", "label": "red gear", "pick_point_2d": [0, 0]},
            {"id": "tray", "kind": "container", "label": "tray", "pick_point_2d": [9, 0]},
        ]}
        spatial = {"centroids_2d": {"anchor": [10, 0], "near": [11, 0], "far": [30, 0], "tray": [10, 0]},
                   "pairwise_distances": [{"a": "anchor", "b": "near", "distance_px": 1}]}
        prompt = prepare_planning_prompt(graph, "pick the gear closest to the mug", 85,
                                         geometry=spatial)
        order = self.payload(prompt, "Computed movable-object distance order from each reference (nearest first):")
        self.assertEqual(order["anchor"], ["near", "far"])
        self.assertEqual(prompt.splitlines()[-2], 'Instruction: "pick the gear closest to the mug"')


if __name__ == "__main__":
    unittest.main()
