"""Test the full stage chain without model weights or network services."""

from contextlib import redirect_stdout, redirect_stderr
import io
import hashlib
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
        self.extra_detections = []

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
            if 'Instruction: "pick the blue cube"' in prompt:
                reply = {"decision": "refuse", "steps": [], "question": None,
                         "reason": "The blue cube is absent from the supplied scene."}
            elif 'Instruction: "throw the red cube"' in prompt:
                reply = {"decision": "refuse", "steps": [], "question": None,
                         "reason": "Throwing is not a controlled pick/place operation."}
            else:
                reply = {"decision": "plan", "steps": [{"action": "pick", "object_id": "red_cube_1"}],
                         "question": None, "reason": None}
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode())

    def detections(self, image_path, queries, box_threshold, text_threshold):
        self.events.append("grounding")
        self.queries.append(queries)
        return [{"id": f"raw_{i}", "label": "red cube", "box": [20, 20, 40, 40], "score": 0.9}
                for i in (1, 2)] + self.extra_detections

    def masks(self, segmenter, image_path, detections):
        self.events.append("segmentation")
        # Actual cleanup must have merged the duplicate boxes before SAM.
        self.assertEqual(len(detections), 1 + len(self.extra_detections))
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
                         "perception_config.json", "run_manifest.json",
                         "planning_policy.txt", "results.json", "results.raw.jsonl"):
            self.assertTrue((self.output / filename).exists(), filename)
        segmentation = json.loads((self.output / "segmentation_results.json").read_text())[0]
        obj = segmentation["objects"][0]
        self.assertNotIn("mask", obj)
        self.assertTrue(Path(obj["mask_path"]).is_file())
        self.assertTrue((self.output / "scene_graphs" / "scene_01.json").is_file())
        results = json.loads(self.out.read_text())["results"]
        self.assertEqual(results[0]["target_pixel"], obj["safe_pixel"])
        self.assertEqual(results[1]["actions"][0]["action"], "refuse")
        self.assertIn("blue cube", results[1]["actions"][0]["reason"])
        self.assertIsNone(results[1]["target_pixel"])
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        policy = (self.output / "planning_policy.txt").read_text()
        policy_hash = hashlib.sha256(policy.encode()).hexdigest()
        self.assertEqual(manifest["planning_policy_sha256"], policy_hash)
        self.assertEqual(Path(manifest["repository_root"]), Path(__file__).resolve().parents[1])
        self.assertEqual(manifest["source_sha256"]["run_grounded.py"],
                         hashlib.sha256((Path(__file__).resolve().parents[1] / "run_grounded.py")
                                        .read_bytes()).hexdigest())
        raw = [json.loads(line) for line in self.out.with_suffix(".raw.jsonl").read_text().splitlines()]
        planning_requests = [body for _, body in self.requests if
                             body.get("response_format", {}).get("json_schema", {}).get("name") == "robot_plan"]
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(planning_requests), 2)
        for entry in raw:
            self.assertEqual(entry["planning_policy_sha256"], policy_hash)
            self.assertEqual(entry["prompt_sha256"], hashlib.sha256(entry["prompt"].encode()).hexdigest())
            self.assertIn(policy.strip(), entry["prompt"])
            self.assertEqual(entry["requests"][0]["messages"], [{"role": "user", "content": entry["prompt"]}])
            self.assertEqual([trace["reply"] for trace in entry["requests"]], entry["replies"])
        planned, absent = raw
        self.assertEqual([trace["stage"] for trace in planned["requests"]], ["draft"])
        self.assertEqual([trace["attempt"] for trace in planned["requests"]], [1])
        for trace, body in zip(planned["requests"], planning_requests):
            self.assertEqual(trace["messages"], body["messages"])
            self.assertEqual(trace["model"], body["model"])
            self.assertEqual(trace["temperature"], body["temperature"])
            self.assertEqual(trace["max_tokens"], body["max_tokens"])
            self.assertTrue(trace["constrained_decoding"])
            self.assertIsNone(trace["validation_error"])
        self.assertEqual([trace["stage"] for trace in absent["requests"]], ["draft"])
        self.assertEqual(absent["requests"][0]["model"], "test-apertus")
        self.assertTrue(absent["requests"][0]["constrained_decoding"])
        self.assertEqual(json.loads(absent["requests"][0]["reply"])["decision"], "refuse")
        for entry in raw:
            self.assertNotIn("Deterministic reference check", entry["prompt"])
            self.assertEqual(len(entry["requests"]), 1)
        self.assertIsNone(absent["error"])

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
        import run_grounded

        args = run_grounded.create_parser().parse_args([])
        self.assertIsNone(args.input)
        self.assertEqual(args.query_source, "qwen_objects")
        self.assertEqual(args.out, Path("outputs/grounded_pipeline/results.json"))
        self.assertEqual(GroundedPerceptionConfig().query_source, "qwen_objects")
    def test_default_run_uses_one_planning_pass_and_logs_its_actual_prompt(self):
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--limit", "1"), 0)
        self.assertEqual(self.events.count("planning"), 1)
        raw = json.loads(self.out.with_suffix(".raw.jsonl").read_text())
        self.assertEqual([trace["stage"] for trace in raw["requests"]], ["draft"])
        body = next(body for _, body in self.requests if
                    body.get("response_format", {}).get("json_schema", {}).get("name") == "robot_plan")
        self.assertEqual(raw["prompt"], body["messages"][0]["content"])
        self.assertNotIn("Deterministic reference check", raw["prompt"])
        self.assertEqual(raw["prompt_sha256"], hashlib.sha256(raw["prompt"].encode()).hexdigest())

    def test_unsafe_requested_operation_is_decided_by_apertus(self):
        (self.data / "scene_01" / "instructions.txt").write_text("secret_category\tthrow the red cube\n")
        self.assertEqual(self.run_pipeline("--scene", "scene_01"), 0)
        results = json.loads(self.out.read_text())["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["instruction_id"], "scene_01_i1")
        self.assertIsNone(results[0]["target_pixel"])
        self.assertEqual(results[0]["actions"][0]["action"], "refuse")
        self.assertIn("Throwing", results[0]["actions"][0]["reason"])
        raw = json.loads(self.out.with_suffix(".raw.jsonl").read_text())
        self.assertEqual([trace["stage"] for trace in raw["requests"]], ["draft"])
        self.assertEqual(json.loads(raw["requests"][0]["reply"])["decision"], "refuse")
        self.assertEqual(raw["requests"][0]["model"], "test-apertus")
        self.assertEqual(self.events.count("planning"), 1)
        self.assertIsNone(raw["error"])
        for _, body in self.requests:
            self.assertNotIn("secret_category", json.dumps(body))

    def test_saved_geometry_preserves_graph_and_skips_upstream_perception(self):
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--build-only"), 0)
        geometry = self.output / "geometry_results.json"
        saved_graph = json.loads((self.output / "scene_graphs" / "scene_01.json").read_text())
        self.requests.clear()
        self.events.clear()
        self.queries.clear()
        self.output = self.root / "replay"
        self.out = self.output / "results.json"
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--geometry-input", str(geometry)), 0)
        self.assertEqual(self.events, ["planning", "planning"])
        self.assertEqual(self.queries, [])
        self.assertTrue(all("apertus.test" in url for url, _ in self.requests))
        self.assertEqual(json.loads((self.output / "scene_graphs" / "scene_01.json").read_text()), saved_graph)


class StageEntrypointTests(unittest.TestCase):
    def test_stage_model_defaults_match_the_full_pipeline(self):
        from run_grounded import create_parser
        from src.apertus import ApertusPlanner
        from src.language.object_extraction import MODEL
        from src.scene_discovery import QWEN_MODEL

        args = create_parser().parse_args([])
        config = GroundedPerceptionConfig()
        self.assertEqual(ApertusPlanner().model, args.apertus_model)
        self.assertEqual(MODEL, args.apertus_model)
        self.assertEqual(config.apertus_model, args.apertus_model)
        self.assertEqual(QWEN_MODEL, args.qwen_model)
        self.assertEqual(config.qwen_model, args.qwen_model)

    def test_extraction_cli_uses_project_schema_or_explicit_override(self):
        from src.language import object_extraction
        from src.schemas import load_object_extraction_schema

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema_path = root / "custom.schema.json"
            override = {"type": "object", "properties": {"objects": {
                "type": "array", "items": {"type": "string", "enum": ["red gear"]},
            }}, "required": ["objects"], "additionalProperties": False}
            schema_path.write_text(json.dumps(override))
            for explicit in (False, True):
                with self.subTest(explicit=explicit):
                    out = root / f"extractions-{explicit}.json"
                    argv = ["object_extraction", "--data", str(root), "--out", str(out)]
                    if explicit:
                        argv.extend(["--schema", str(schema_path)])
                    with patch("sys.argv", argv), \
                            patch.object(object_extraction, "load_instructions", return_value=[{
                                "scene": "scene_01", "category": "simple", "instruction": "pick the red gear",
                            }]), \
                            patch.object(object_extraction, "ask_apertus", return_value={"objects": ["red gear"]}) as ask, \
                            redirect_stdout(io.StringIO()):
                        object_extraction.main()
                    self.assertEqual(ask.call_args.kwargs["schema"],
                                     override if explicit else load_object_extraction_schema())
                    self.assertEqual(json.loads(out.read_text())[0]["extracted_objects"], ["red gear"])


if __name__ == "__main__":
    unittest.main()
