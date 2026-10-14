"""Check planning inputs from the sole grounded scene graph format."""

from copy import deepcopy
import json
import re
import unittest

from src.prompting import derived_geometry, prepare_planning_prompt


class PlanningPromptTests(unittest.TestCase):
    def graph(self):
        return {"image": {"width": 100, "height": 80}, "objects": [
            {"id": "red_gear_1", "label": "red gear", "kind": "movable", "color": "red",
             "pick_point_2d": [10, 20], "width_mm": 30, "graspable": True,
             "bbox_2d": [0, 0, 30, 40], "confidence": 0.9,
             "relations": {"inside": "grey_tray_1", "on_top_of": None, "supports": [],
                           "held_by_human": False, "partially_occluded": False}},
            {"id": "grey_tray_1", "label": "grey tray", "kind": "container", "color": "grey",
             "pick_point_2d": [40, 60], "width_mm": 200, "graspable": False,
             "bbox_2d": [0, 0, 99, 79], "confidence": 0.8,
             "relations": {"inside": None, "on_top_of": None, "supports": [],
                           "held_by_human": False, "partially_occluded": False}},
        ]}

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
        graph = self.graph()
        before = deepcopy(graph)
        instruction = 'put the red gear in the grey tray, not "that mug"'
        prompt = prepare_planning_prompt(graph, instruction, 85)
        self.assertEqual(self.payload(prompt, "Scene graph:"), graph)
        self.assertEqual(graph, before)
        self.assertIn("Instruction: " + json.dumps(instruction), prompt)
        self.assert_field_guidance_exists_in_graph(prompt, graph)
        for absent in ("target_pixel", "needs_review", "review_reasons"):
            self.assertNotIn(absent, prompt)
        geometry = self.payload(prompt, "Deterministically derived 2D geometry:")
        self.assertEqual(geometry["left_to_right"], ["red_gear_1", "grey_tray_1"])
        self.assertEqual(geometry["pairwise_distances"], [{"a": "red_gear_1", "b": "grey_tray_1", "distance_px": 50.0}])

    def test_optional_measurements_are_not_described_when_missing(self):
        graph = self.graph()
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
        graph = {"image": {"width": 100, "height": 80}, "objects": []}
        prompt = prepare_planning_prompt(graph, "pick the blue cube", 85)
        self.assertEqual(self.payload(prompt, "Scene graph:"), graph)
        self.assertEqual(self.payload(prompt, "Deterministically derived 2D geometry:")["pairwise_distances"], [])

    def test_multiple_visual_supports_preserve_ids_and_provenance(self):
        graph = self.graph()
        upper = deepcopy(graph["objects"][0])
        upper.update(id="red_cube_1", label="red cube")
        upper["relations"].update(inside=None, on_top_of=["red_gear_1", "grey_tray_1"])
        graph["objects"].append(upper)
        for obj in graph["objects"][:2]:
            obj["relations"].update(on_top_of=[], supports=["red_cube_1"])
        spatial = {"support_source": "qwen", "centroids_2d": {},
                   "pairwise_distances": []}
        before = deepcopy((graph, spatial))
        prompt = prepare_planning_prompt(graph, "move the cube into the tray", 85,
                                         geometry=spatial)
        self.assertEqual(self.payload(prompt, "Scene graph:"), graph)
        self.assertEqual(self.payload(prompt, "Deterministically derived 2D geometry:"), spatial)
        self.assertEqual((graph, spatial), before)
        self.assert_field_guidance_exists_in_graph(prompt, graph)
        self.assertIn("- support_source=qwen:", prompt)
        self.assertNotIn("- support_source=geometry:", prompt)

    def test_support_provenance_guidance_matches_supplied_context(self):
        graph = self.graph()
        for source in ("geometry", "none", None):
            spatial = {} if source is None else {"support_source": source}
            prompt = prepare_planning_prompt(graph, "pick the gear", 85, geometry=spatial)
            provenance = re.findall(r"^- support_source=([a-z]+):", prompt, re.MULTILINE)
            self.assertEqual(provenance, [] if source is None else [source])

    def test_distance_order_uses_centroids_instead_of_pick_points(self):
        graph = {"image": {"width": 120, "height": 80}, "objects": [
            {"id": oid, "kind": kind, "label": label, "color": None,
             "pick_point_2d": point, "bbox_2d": [0, 0, 119, 79], "confidence": 0.9,
             "relations": {"inside": None, "on_top_of": None, "supports": [],
                           "held_by_human": False, "partially_occluded": False}}
            for oid, kind, label, point in (
                ("anchor", "movable", "mug", [10, 0]),
                ("near", "movable", "yellow gear", [100, 0]),
                ("far", "movable", "red gear", [0, 0]),
                ("tray", "container", "tray", [9, 0]),
            )
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
