"""Test the full stage chain without model weights or network services."""

from contextlib import redirect_stdout, redirect_stderr
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
from src.grounded_perception import GroundedPerceptionConfig
from src.masking.sam import SamSegmenter


class GroundedPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "dataset"
        self.data.mkdir()
        (self.data / "calibration.json").write_text(json.dumps({
            "camera": {"intrinsics": {"K": np.eye(3).tolist()}},
            "gripper_max_opening_mm": 85,
        }))
        for name in ("scene_01", "scene_02"):
            folder = self.data / name
            folder.mkdir()
            cv2.imwrite(str(folder / "rgb.png"), np.zeros((80, 120, 3), np.uint8))
            (folder / "instructions.txt").write_text(
                "secret_category\tpick the red cube\nsecret_category\tpick the blue cube\n")
        self.output = self.root / "outputs"
        self.out = self.output / "results.json"
        self.requests = []
        self.queries = []
        self.events = []
        self.discovery_replies = 0

    def http_response(self, request, **kwargs):
        body = json.loads(request.data)
        self.requests.append((request.full_url, body))
        prompt = body["messages"][0]["content"]
        if isinstance(prompt, str) and "Extract every explicitly mentioned" in prompt:
            self.events.append("extraction")
            reply = {"objects": ["blue cube" if "Instruction: pick the blue cube" in prompt else "red cube"]}
        elif isinstance(prompt, list):
            self.events.append("discovery")
            # Verification removes a spurious first-pass category.
            self.discovery_replies += 1
            reply = {"objects": ["red cube", "mug"] if self.discovery_replies % 2 else ["red cube"]}
        else:
            self.events.append("planning")
            reply = {"decision": "plan", "steps": [{"action": "pick", "object_id": "red_cube_1"}],
                     "question": None, "reason": None}
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode())

    def detections(self, image_path, queries, box_threshold, text_threshold):
        self.events.append("grounding")
        self.queries.append(queries)
        return [{"id": f"raw_{i}", "label": "red cube", "box": [20, 20, 40, 40], "score": 0.9}
                for i in (1, 2)]

    def masks(self, segmenter, image_path, detections):
        self.events.append("segmentation")
        # Actual cleanup must have merged the duplicate boxes before SAM.
        self.assertEqual(len(detections), 1)
        shape = cv2.imread(str(image_path)).shape[:2]
        objects = []
        for detection in detections:
            mask = np.zeros(shape, bool)
            x1, y1, x2, y2 = map(int, detection["box"])
            mask[y1:y2, x1:x2] = True
            objects.append(segmenter.segment_object(detection, mask[None], np.array([0.95])))
        return objects

    def run_pipeline(self, *extra):
        def segment(segmenter, image_path, detections):
            return self.masks(segmenter, image_path, detections)

        with patch("urllib.request.urlopen", side_effect=self.http_response), \
                patch("src.grounding.grounding_dino.GroundingDinoDetector") as backend, \
                patch.object(SamSegmenter, "segment_scene", new=segment), \
                redirect_stdout(io.StringIO()):
            backend.return_value.detect.side_effect = self.detections
            result = main(["--data", str(self.data), "--out", str(self.out),
                           "--apertus-base-url", "http://apertus.test/v1",
                           "--apertus-model", "test-apertus",
                           "--qwen-base-url", "http://qwen.test/v1", "--qwen-model", "test-qwen",
                           *extra])
        return result

    def test_full_entrypoint_hands_verified_labels_to_dino_and_saves_all_stages(self):
        self.assertEqual(self.run_pipeline("--scene", "scene_01"), 0)
        self.assertEqual(self.events, ["extraction", "extraction", "discovery", "discovery",
                                       "grounding", "segmentation", "planning", "planning"])
        self.assertEqual(self.queries, [["red cube"]])
        # The absent blue cube goes to Qwen as a hint, not directly to DINO.
        discovery_prompt = self.requests[2][1]["messages"][0]["content"][1]["text"]
        self.assertIn('"blue cube"', discovery_prompt)
        self.assertIn('"red cube"', discovery_prompt)
        for _, body in self.requests:
            self.assertNotIn("secret_category", json.dumps(body))
        for url, body in self.requests:
            expected = "test-qwen" if "qwen.test" in url else "test-apertus"
            self.assertEqual(body["model"], expected)
        for filename in ("object_extractions.json", "scene_discovery.json", "grounding_detections.json",
                         "grounding_detections_clean.json", "grounding_cleanup_report.json",
                         "segmentation_results.json", "geometry_results.json", "geometry_debug.json",
                         "perception_config.json", "results.json", "results.raw.jsonl"):
            self.assertTrue((self.output / filename).exists(), filename)
        segmentation = json.loads((self.output / "segmentation_results.json").read_text())[0]
        obj = segmentation["objects"][0]
        self.assertNotIn("mask", obj)
        self.assertTrue(Path(obj["mask_path"]).is_file())
        self.assertTrue((self.output / "scene_graphs" / "scene_01.json").is_file())
        result = json.loads(self.out.read_text())["results"][0]
        self.assertEqual(result["target_pixel"], obj["safe_pixel"])

    def test_limit_skips_later_scenes_but_extracts_all_selected_scene_instructions(self):
        self.run_pipeline("--limit", "1")
        extractions = json.loads((self.output / "object_extractions.json").read_text())
        self.assertEqual(len(extractions), 2)
        self.assertEqual({e["scene"] for e in extractions}, {"scene_01"})
        self.assertEqual(len(json.loads(self.out.read_text())["results"]), 1)
        self.assertFalse((self.output / "scene_graphs" / "scene_02.json").exists())

    def test_combined_queries_are_an_explicit_option(self):
        self.run_pipeline("--scene", "scene_01", "--query-source", "grounding_queries")
        self.assertEqual(self.queries, [["blue cube", "red cube"]])

    def test_fresh_build_only_runs_perception_and_skips_planning(self):
        self.run_pipeline("--scene", "scene_01", "--build-only")
        self.assertEqual(self.events.count("extraction"), 2)
        self.assertNotIn("planning", self.events)
        self.assertFalse(self.out.exists())
        self.assertTrue((self.output / "scene_graphs" / "scene_01.json").exists())

    def test_invalid_qwen_output_stops_before_dino_and_records_stage(self):
        original = self.http_response

        def invalid_qwen(request, **kwargs):
            if "qwen.test" in request.full_url:
                return io.BytesIO(json.dumps({"choices": [{"message": {
                    "content": json.dumps({"objects": [123]})}}]}).encode())
            return original(request, **kwargs)

        self.http_response = invalid_qwen
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_pipeline("--scene", "scene_01")
        failure = json.loads((self.output / "perception.failure.json").read_text())
        self.assertEqual(failure["stage"], "scene_discovery")
        self.assertEqual(failure["scene"], "scene_01")
        self.assertEqual(self.queries, [])
        self.assertFalse(self.out.exists())

    def test_connection_failure_stops_during_extraction(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("unreachable")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            main(["--data", str(self.data), "--out", str(self.out)])
        failure = json.loads((self.output / "perception.failure.json").read_text())
        self.assertEqual(failure["stage"], "object_extraction")
        self.assertFalse(self.out.exists())

    def test_default_entrypoint_runs_full_pipeline(self):
        import run_geometry
        import run_grounded

        self.assertIs(run_geometry.main, run_grounded.main)
        args = run_grounded.create_parser().parse_args([])
        self.assertIsNone(args.input)
        self.assertEqual(args.query_source, "qwen_objects")
        self.assertEqual(args.out, Path("outputs/grounded_pipeline/results.json"))
        self.assertEqual(GroundedPerceptionConfig().query_source, "qwen_objects")


if __name__ == "__main__":
    unittest.main()
