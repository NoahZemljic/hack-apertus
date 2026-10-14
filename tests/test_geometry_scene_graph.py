"""Verify the geometry-to-planner boundary using real masks and mocked HTTP."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import cv2
import numpy as np

from run_grounded import main
from src.apertus import ApertusPlanner, PlanningGenerationError
from src.geometry import GeometryReasoner
from src.geometry_scene_graph import GeometrySceneGraphBuilder
from src.grounding.postprocess_grounding import process_scene
from src.masking.sam import SamSegmenter
from src.prompting import prepare_planning_prompt
from src.validation import validate_scene_graph


class GeometrySceneGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scene = {"scene": "scene_test", "image": {"width": 100, "height": 60},
                      "geometry_version": "1.0", "objects": [],
                      "relations": [], "relation_candidates": []}
        self.input = self.root / "geometry.json"

    def object(self, oid, label, bounds, *, hole=False):
        x1, y1, x2, y2 = bounds
        mask = np.zeros((60, 100), np.uint8)
        mask[y1:y2, x1:x2] = 255
        if hole:
            mask[y1+3:y2-3, x1+3:x2-3] = 0
        ys, xs = np.nonzero(mask)
        path = f"{oid}.png"
        cv2.imwrite(str(self.root / path), mask)
        obj = {"id": oid, "label": label, "aliases": [label], "box": list(bounds),
               "mask_box": list(bounds), "mask_area": int(xs.size), "mask_path": path,
               "centroid": [float(xs.mean()), float(ys.mean())],
               "safe_pixel": [int(xs[0]), int(ys[0])], "score": 0.8,
               "needs_review": True, "review_reasons": ["test diagnostic"]}
        self.scene["objects"].append(obj)
        return obj

    def edge(self, source, relation, target, *, candidate=False):
        self.scene["relation_candidates" if candidate else "relations"].append({
            "source": source, "type": relation, "target": target, "needs_review": True,
            "scope": "physical_hypothesis" if candidate else "image_plane",
        })

    def builder(self, **kwargs):
        self.input.write_text(json.dumps([self.scene]))
        return GeometrySceneGraphBuilder(self.input, **kwargs)

    def test_image_border_ring_preserves_foreground_pick_and_inclusive_box(self):
        obj = self.object("gear_1", "yellow gear", (80, 40, 100, 60), hole=True)
        graph, _, _ = self.builder().build("scene_test", 100, 60)
        self.assertEqual(graph["objects"][0]["bbox_2d"], [80, 40, 99, 59])
        self.assertEqual(graph["objects"][0]["pick_point_2d"], obj["safe_pixel"])
        self.assertNotEqual(obj["safe_pixel"], obj["centroid"])
        validate_scene_graph(graph, 100, 60)

    def test_graph_detector_confidence_prefers_explicit_null_with_legacy_fallbacks(self):
        unknown = self.object("gear_1", "red gear", (10, 10, 30, 30))
        unknown.update(detector_score=None, dino_score=0.8, sam_score=0.99, backend="qwen")
        legacy = self.object("gear_2", "yellow gear", (50, 10, 70, 30))
        legacy["dino_score"] = 0.7
        graph, _, _ = self.builder().build("scene_test", 100, 60)
        self.assertIsNone(graph["objects"][0]["confidence"])
        self.assertEqual(graph["objects"][1]["confidence"], 0.7)
        validate_scene_graph(graph, 100, 60)

    def test_qwen_null_confidence_survives_cleanup_sam_geometry_and_graph(self):
        image_path = self.root / "rgb.png"
        cv2.imwrite(str(image_path), np.zeros((60, 100, 3), np.uint8))
        raw = {"scene": "scene_test", "backend": "qwen", "queries": ["red gear"], "detections": [
            {"id": "right", "label": "red gear", "box": [50, 10, 70, 30], "score": None},
            {"id": "left", "label": "red gear", "box": [10, 10, 30, 30], "score": None},
        ]}
        cleaned, _ = process_scene(raw, image_path)
        clean_path = self.root / "clean.json"
        clean_path.write_text(json.dumps([cleaned]))
        segmenter = SamSegmenter()
        objects = []
        for detection in segmenter.load_detections(clean_path)[0]["detections"]:
            mask = np.zeros((60, 100), bool)
            x1, y1, x2, y2 = map(int, detection["box"])
            mask[y1:y2, x1:x2] = True
            segmented = segmenter.segment_object(detection, mask[None], np.array([0.99]))
            segmented["mask_path"] = str(segmenter.save_mask(mask, self.root / f"{detection['id']}.png"))
            objects.append({k: v for k, v in segmented.items() if k != "mask"})
        geometry, _ = GeometryReasoner(base_dir=self.root).analyze_scene({
            "scene": "scene_test", "image_path": str(image_path), "objects": objects,
        })
        self.input.write_text(json.dumps([geometry]))
        graph, _, _ = GeometrySceneGraphBuilder(self.input).build("scene_test", 100, 60)
        self.assertEqual([obj["id"] for obj in graph["objects"]], ["red_gear_1", "red_gear_2"])
        self.assertTrue(all(obj["confidence"] is None for obj in graph["objects"]))
        self.assertTrue(all(obj["backend"] == "qwen" and obj["detector_score"] is None for obj in geometry["objects"]))
        self.assertTrue(all(obj["sam_score"] == 0.99 and "dino_score" not in obj for obj in geometry["objects"]))
        self.assertEqual([obj["pick_point_2d"] for obj in graph["objects"]],
                         [obj["safe_pixel"] for obj in objects])
        validate_scene_graph(graph, 100, 60)

    def test_membership_requires_container_and_does_not_create_support(self):
        self.object("tray_1", "grey tray", (0, 0, 70, 50))
        self.object("gear_1", "red gear", (10, 10, 30, 30))
        self.object("gear_2", "yellow gear", (15, 15, 25, 25))
        self.edge("gear_1", "inside", "tray_1")
        self.edge("gear_2", "inside", "gear_1")
        graph, _, debug = self.builder().build("scene_test", 100, 60)
        objects = {o["id"]: o for o in graph["objects"]}
        self.assertEqual(objects["gear_1"]["relations"]["inside"], "tray_1")
        self.assertIsNone(objects["gear_2"]["relations"]["inside"])
        self.assertEqual(objects["tray_1"]["relations"]["supports"], [])
        self.assertEqual(len(debug["rejected_relations"]), 1)

    def test_selected_support_has_inverse_and_can_be_disabled(self):
        self.object("gear_1", "red gear", (10, 10, 30, 30))
        self.object("gear_2", "yellow gear", (15, 15, 25, 25))
        self.edge("gear_2", "on", "gear_1", candidate=True)
        graph, spatial, debug = self.builder().build("scene_test", 100, 60)
        self.assertEqual(graph["objects"][1]["relations"]["on_top_of"], ["gear_1"])
        self.assertEqual(graph["objects"][0]["relations"]["supports"], ["gear_2"])
        self.assertEqual(spatial["support_source"], "geometry")
        self.assertEqual(debug["accepted_relations"][0]["evidence_source"], "geometry")
        graph, spatial, _ = self.builder(accept_support_candidates=False).build("scene_test", 100, 60)
        self.assertEqual(graph["objects"][1]["relations"]["on_top_of"], [])
        self.assertEqual(spatial["support_source"], "none")

    def test_visual_supports_override_wrong_geometry_and_allow_two_supporting_gears(self):
        self.object("gear_1", "red gear", (10, 20, 35, 45))
        self.object("gear_2", "red gear", (40, 20, 65, 45))
        self.object("cube_1", "red cube", (25, 15, 50, 30))
        self.object("tray_1", "grey tray", (0, 0, 75, 55))
        self.edge("gear_1", "on", "cube_1", candidate=True)
        self.edge("cube_1", "inside", "tray_1")
        self.edge("gear_1", "left_of", "gear_2")
        self.scene["visual_supports"] = {"backend": "qwen", "supports": [
            {"upper": "cube_1", "lower": "gear_1"},
            {"upper": "cube_1", "lower": "gear_2"},
        ]}
        builder = self.builder(accept_support_candidates=False)
        original = deepcopy(builder.records)
        graph, spatial, debug = builder.build("scene_test", 100, 60)
        objects = {obj["id"]: obj for obj in graph["objects"]}
        self.assertEqual(objects["cube_1"]["relations"]["on_top_of"], ["gear_1", "gear_2"])
        self.assertEqual(objects["cube_1"]["relations"]["inside"], "tray_1")
        for oid in ("gear_1", "gear_2"):
            self.assertEqual(objects[oid]["relations"]["supports"], ["cube_1"])
            self.assertEqual(objects[oid]["relations"]["on_top_of"], [])
        self.assertEqual(spatial["support_source"], "qwen")
        self.assertEqual(spatial["image_plane_relations"], [
            {"source": "gear_1", "type": "left_of", "target": "gear_2"}])
        self.assertTrue(all(edge["evidence_source"] == "visual"
                            for edge in debug["accepted_relations"] if edge["type"] == "on"))
        self.assertIn("override", debug["rejected_relations"][0]["builder_reason"])
        self.assertEqual(builder.records, original)
        validate_scene_graph(graph, 100, 60)

    def test_empty_visual_observation_prevents_geometry_false_support(self):
        self.object("gear_1", "red gear", (10, 10, 30, 30))
        self.object("gear_2", "yellow gear", (15, 15, 25, 25))
        self.edge("gear_2", "on", "gear_1", candidate=True)
        self.scene["visual_supports"] = {"backend": "qwen", "supports": []}
        graph, spatial, debug = self.builder().build("scene_test", 100, 60)
        self.assertTrue(all(obj["relations"]["on_top_of"] == [] for obj in graph["objects"]))
        self.assertTrue(all(obj["relations"]["supports"] == [] for obj in graph["objects"]))
        self.assertEqual(spatial["support_source"], "qwen")
        self.assertEqual(debug["input_visual_supports"], self.scene["visual_supports"])
        self.assertEqual(debug["rejected_relations"][0]["evidence_source"], "geometry")

    def test_visual_supports_reject_bad_references_cycles_and_non_workpiece_edges(self):
        self.object("gear_1", "red gear", (10, 10, 30, 30))
        self.object("gear_2", "yellow gear", (15, 15, 25, 25))
        self.object("tray_1", "grey tray", (0, 0, 75, 55))
        self.object("hand_1", "hand", (50, 10, 75, 30))
        cases = [
            ([{"upper": "gear_1", "lower": "missing"}], "unknown object"),
            ([{"upper": "gear_1", "lower": "gear_1"}], "self"),
            ([{"upper": "gear_1", "lower": "gear_2"},
              {"upper": "gear_2", "lower": "gear_1"}], "cycle"),
            ([{"upper": "gear_1", "lower": "gear_2"}] * 2, "duplicate"),
        ]
        for invalid_kind in ("tray_1", "hand_1"):
            cases.extend((
                ([{"upper": invalid_kind, "lower": "gear_1"}], "upper"),
                ([{"upper": "gear_1", "lower": invalid_kind}], "lower"),
            ))
        for supports, message in cases:
            self.scene["visual_supports"] = {"backend": "qwen", "supports": supports}
            with self.subTest(supports=supports), self.assertRaisesRegex(ValueError, message):
                self.builder().build("scene_test", 100, 60)

    def test_visual_support_allows_a_plate_as_upper_object(self):
        self.object("gear_1", "red gear", (10, 20, 35, 45))
        self.object("plate_1", "white plate", (10, 10, 35, 30))
        self.scene["visual_supports"] = {"backend": "qwen", "supports": [
            {"upper": "plate_1", "lower": "gear_1"},
        ]}
        graph, _, _ = self.builder().build("scene_test", 100, 60)
        self.assertEqual(graph["objects"][1]["kind"], "surface")
        self.assertEqual(graph["objects"][1]["relations"]["on_top_of"], ["gear_1"])

    def test_malformed_visual_support_contract_fails_instead_of_falling_back(self):
        self.object("gear_1", "red gear", (10, 10, 30, 30))
        for invalid in (None, [], {"backend": "geometry", "supports": []},
                        {"backend": "qwen", "supports": None},
                        {"backend": "qwen", "supports": [{"upper": "gear_1"}]},
                        {"backend": "qwen", "supports": [], "extra": True}):
            self.scene["visual_supports"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "visual[ _]support"):
                self.builder().build("scene_test", 100, 60)

    def test_debug_is_separate_and_spatial_distances_use_centroids(self):
        a = self.object("part_1", "white part", (0, 0, 40, 20))
        b = self.object("part_2", "white part", (60, 0, 70, 10))
        self.edge("part_1", "left_of", "part_2")
        builder = self.builder()
        original = deepcopy(builder.records)
        graph, spatial, debug = builder.build("scene_test", 100, 60)
        self.assertEqual(builder.records, original)
        prompt = prepare_planning_prompt(graph, "pick the left part", None, geometry=spatial)
        for field in ("needs_review", "review_reasons", "physical_hypothesis"):
            self.assertNotIn(field, prompt)
        self.assertTrue(debug["objects"][0]["needs_review"])
        distance = np.linalg.norm(np.array(a["centroid"]) - b["centroid"])
        self.assertEqual(spatial["pairwise_distances"][0]["distance_px"], round(float(distance), 1))

    def test_bad_mask_metadata_and_unknown_relation_targets_fail(self):
        obj = self.object("cube_1", "red cube", (10, 10, 30, 30))
        obj["safe_pixel"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "safe_pixel"):
            self.builder().build("scene_test", 100, 60)
        obj["safe_pixel"] = [10, 10]
        self.edge("cube_1", "inside", "missing_tray")
        with self.assertRaisesRegex(ValueError, "unknown object"):
            self.builder().build("scene_test", 100, 60)

    def test_cycles_are_not_emitted(self):
        self.object("gear_1", "red gear", (10, 10, 30, 30))
        self.object("gear_2", "yellow gear", (15, 15, 25, 25))
        self.edge("gear_1", "on", "gear_2", candidate=True)
        self.edge("gear_2", "on", "gear_1", candidate=True)
        graph, _, debug = self.builder().build("scene_test", 100, 60)
        validate_scene_graph(graph, 100, 60)
        self.assertTrue(any("cycle" in e["builder_reason"] for e in debug["rejected_relations"]))

    def dataset(self):
        data = self.root / "dataset"
        folder = data / "scene_test"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "rgb.png"), np.zeros((60, 100, 3), np.uint8))
        (folder / "instructions.txt").write_text("simple\tpick the red cube\n")
        (data / "calibration.json").write_text(json.dumps({
            "camera": {"intrinsics": {"K": np.eye(3).tolist()}}, "gripper_max_opening_mm": 85,
        }))
        self.object("cube_1", "red cube", (10, 10, 30, 30))
        self.builder()
        return data

    @staticmethod
    def response(plan):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(plan)}}]}).encode())

    def test_cli_http_request_and_submission_target(self):
        data, out = self.dataset(), self.root / "outputs" / "results.json"
        plan = {"decision": "plan", "steps": [{"action": "pick", "object_id": "cube_1"}],
                "question": None, "reason": None}
        requests = []

        def respond(request, **kwargs):
            requests.append(json.loads(request.data))
            return self.response(plan)

        with patch("urllib.request.urlopen", side_effect=respond), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--input", str(self.input), "--data", str(data), "--out", str(out)]), 0)
        entry = json.loads(out.read_text())["results"][0]
        self.assertEqual(entry["instruction_id"], "scene_test_i1")
        self.assertEqual(entry["target_pixel"], [10, 10])
        self.assertEqual(entry["actions"], [{"action": "pick", "object": "cube_1"}])
        self.assertIn("response_format", requests[0])
        prompt = requests[0]["messages"][0]["content"]
        self.assertIn("mask centroids", prompt)
        self.assertNotIn("needs_review", prompt)
        self.assertNotIn('"category":', prompt)

    def test_build_only_preserves_submission_and_makes_no_requests(self):
        data, out = self.dataset(), self.root / "results.json"
        out.write_text("existing results")
        with patch("urllib.request.urlopen") as server, redirect_stdout(io.StringIO()):
            main(["--input", str(self.input), "--data", str(data), "--out", str(out), "--build-only"])
        server.assert_not_called()
        self.assertEqual(out.read_text(), "existing results")

    def test_unreachable_server_does_not_write_fake_refusals(self):
        data, out = self.dataset(), self.root / "outputs" / "results.json"
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("unreachable")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--input", str(self.input), "--data", str(data), "--out", str(out)])
        self.assertEqual(error.exception.code, 1)
        self.assertFalse(out.exists())
        self.assertIn("unreachable", out.with_suffix(".raw.jsonl").read_text())

    def test_format_invalid_planner_replies_survive_retry(self):
        self.object("cube_1", "red cube", (10, 10, 30, 30))
        graph, _, _ = self.builder().build("scene_test", 100, 60)
        invalid = {"decision": "plan", "steps": [{"action": "pick"}],
                   "question": None, "reason": None}
        with patch("urllib.request.urlopen", side_effect=lambda *a, **k: self.response(invalid)):
            with self.assertRaises(PlanningGenerationError) as error:
                ApertusPlanner().plan(graph, "pick the cube", None)
        self.assertEqual(len(error.exception.replies), 2)


if __name__ == "__main__":
    unittest.main()
