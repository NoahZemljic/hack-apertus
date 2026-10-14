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
from src.geometry_scene_graph import GeometrySceneGraphBuilder
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
        graph, _, _ = self.builder().build("scene_test", 100, 60)
        self.assertEqual(graph["objects"][1]["relations"]["on_top_of"], "gear_1")
        self.assertEqual(graph["objects"][0]["relations"]["supports"], ["gear_2"])
        graph, _, _ = self.builder(accept_support_candidates=False).build("scene_test", 100, 60)
        self.assertIsNone(graph["objects"][1]["relations"]["on_top_of"])

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
