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
        self.expected_clean_count = None
        self.stacking_reply = {"supports": []}

    def http_response(self, request, **kwargs):
        body = json.loads(request.data)
        self.requests.append((request.full_url, body))
        prompt = body["messages"][0]["content"]
        if body.get("response_format", {}).get("json_schema", {}).get("name") == "qwen_stacking":
            self.events.append("stacking")
            reply = self.stacking_reply
        elif isinstance(prompt, str) and "Extract every explicitly mentioned" in prompt:
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
        expected = 1 + len(self.extra_detections) if self.expected_clean_count is None else self.expected_clean_count
        self.assertEqual(len(detections), expected)
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
            self.detector_backend = backend
            backend.return_value.detect.side_effect = self.detections
            result = main(["--data", str(self.data), "--out", str(self.out),
                           "--apertus-base-url", "http://apertus.test/v1",
                           "--apertus-model", "test-apertus",
                           "--qwen-base-url", "http://qwen.test/v1", "--qwen-model", "test-qwen",
                           *extra])
        return result

    def use_qwen_boxes(self, reply):
        original = self.http_response

        def boxes_response(request, **kwargs):
            if "qwen.test" not in request.full_url:
                return original(request, **kwargs)
            body = json.loads(request.data)
            if body.get("response_format", {}).get("json_schema", {}).get("name") == "qwen_stacking":
                return original(request, **kwargs)
            self.requests.append((request.full_url, body))
            self.events.append("qwen_boxes")
            return io.BytesIO(json.dumps({"choices": [{"message": {
                "content": json.dumps(reply)}}]}).encode())

        self.http_response = boxes_response

    def use_stacked_objects(self):
        self.use_qwen_boxes({"objects": [
            {"label": "red_cube", "bbox_2d": [350, 200, 650, 500]},
            {"label": "red_gear", "bbox_2d": [200, 350, 450, 650]},
            {"label": "red_gear", "bbox_2d": [550, 350, 800, 650]},
        ]})
        self.expected_clean_count = 3
        self.stacking_reply = {"supports": [
            {"upper": "red_cube_1", "lower": "red_gear_1"},
            {"upper": "red_cube_1", "lower": "red_gear_2"},
        ]}

    def test_separate_stacking_call_enriches_graph_without_changing_objects(self):
        self.use_stacked_objects()
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--box-detector", "qwen"), 0)
        self.assertEqual(self.events, ["extraction", "extraction", "qwen_boxes", "segmentation",
                                       "stacking", "planning", "planning"])
        self.detector_backend.assert_not_called()
        graph = json.loads((self.output / "scene_graphs/scene_01.json").read_text())
        objects = {obj["id"]: obj for obj in graph["objects"]}
        self.assertEqual(objects["red_cube_1"]["relations"]["on_top_of"], ["red_gear_1", "red_gear_2"])
        for gear in ("red_gear_1", "red_gear_2"):
            self.assertEqual(objects[gear]["relations"]["supports"], ["red_cube_1"])
        segmentation = json.loads((self.output / "segmentation_results.json").read_text())[0]
        for obj in segmentation["objects"]:
            self.assertEqual(objects[obj["id"]]["pick_point_2d"], obj["safe_pixel"])
        trace = json.loads((self.output / "qwen_stacking.raw.json").read_text())[0]
        self.assertEqual(len(trace["requests"]), 1)
        self.assertEqual(trace["errors"], [])
        self.assertNotIn("data:image", json.dumps(trace))
        request = next(body for _, body in self.requests if
                       body.get("response_format", {}).get("json_schema", {}).get("name") == "qwen_stacking")
        self.assertNotIn("pick the red cube", json.dumps(request))
        spatial = json.loads((self.output / "scene_graphs/scene_01.spatial.json").read_text())
        self.assertEqual(spatial["support_source"], "qwen")

    def prepare_geometry_replay(self):
        self.use_stacked_objects()
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--box-detector", "qwen",
                                           "--stacking-source", "geometry", "--build-only"), 0)
        geometry = self.output / "geometry_results.json"
        records = json.loads(geometry.read_text())
        # A deliberately reversed legacy estimate must be replaced by the visual
        # task, using the same object IDs and saved foreground masks.
        records[0]["relation_candidates"] = [{"source": "red_gear_1", "target": "red_cube_1", "type": "on"}]
        for obj in records[0]["objects"]:
            obj["mask_path"] = str(Path(obj["mask_path"]).relative_to(self.output))
        geometry.write_text(json.dumps(records))
        original = geometry.read_bytes()
        self.requests.clear()
        self.events.clear()
        self.queries.clear()
        self.output = self.root / "stacking-replay"
        self.out = self.output / "results.json"
        return geometry, original

    def test_stacking_refresh_reuses_geometry_and_masks_and_saves_new_checkpoint(self):
        geometry, original = self.prepare_geometry_replay()
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--geometry-input", str(geometry),
                                           "--refresh-stacking"), 0)
        self.assertEqual(self.events, ["stacking", "planning", "planning"])
        self.detector_backend.assert_not_called()
        self.assertEqual(geometry.read_bytes(), original)
        saved = json.loads((self.output / "geometry_results.json").read_text())[0]
        previous = json.loads(original)[0]
        for obj, old in zip(saved["objects"], previous["objects"]):
            for key in ("box", "safe_pixel", "centroid", "mask_area", "label", "id"):
                self.assertEqual(obj[key], old[key])
            self.assertTrue(Path(obj["mask_path"]).is_file())
        graph = json.loads((self.output / "scene_graphs/scene_01.json").read_text())
        self.assertEqual(graph["objects"][0]["relations"]["on_top_of"], ["red_gear_1", "red_gear_2"])
        self.assertEqual(graph["objects"][1]["relations"]["on_top_of"], [])

    def test_empty_visual_stacking_overrides_previous_geometry_guess(self):
        geometry, _ = self.prepare_geometry_replay()
        self.stacking_reply = {"supports": []}
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--geometry-input", str(geometry),
                                           "--refresh-stacking", "--build-only"), 0)
        self.assertEqual(self.events, ["stacking"])
        graph = json.loads((self.output / "scene_graphs/scene_01.json").read_text())
        self.assertTrue(all(obj["relations"]["on_top_of"] == [] for obj in graph["objects"]))
        self.assertTrue(all(obj["relations"]["supports"] == [] for obj in graph["objects"]))

    def test_stacking_refresh_can_reuse_copied_masks_when_scratch_mount_is_unavailable(self):
        geometry, _ = self.prepare_geometry_replay()
        records = json.loads(geometry.read_text())
        for obj in records[0]["objects"]:
            obj["mask_path"] = str(self.root / "unavailable-scratch" / "segmentation" /
                                   records[0]["scene"] / f"{obj['id']}.png")
        geometry.write_text(json.dumps(records))
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--geometry-input", str(geometry),
                                           "--refresh-stacking", "--build-only"), 0)
        self.assertEqual(self.events, ["stacking"])
        saved = json.loads((self.output / "geometry_results.json").read_text())[0]
        for obj in saved["objects"]:
            self.assertEqual(Path(obj["mask_path"]), geometry.parent / "segmentation" /
                             saved["scene"] / f"{obj['id']}.png")

    def test_bad_stacking_reply_stops_before_planning_and_preserves_failed_trace(self):
        geometry, original = self.prepare_geometry_replay()
        self.stacking_reply = {"supports": [{"upper": "missing", "lower": "red_gear_1"}]}
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_pipeline("--scene", "scene_01", "--geometry-input", str(geometry),
                              "--refresh-stacking", "--qwen-stacking-retries", "0")
        self.assertEqual(self.events, ["stacking"])
        self.assertEqual(geometry.read_bytes(), original)
        self.assertFalse(self.out.exists())
        failure = json.loads((self.output / "perception.failure.json").read_text())
        self.assertEqual((failure["stage"], failure["scene"]), ("qwen_stacking", "scene_01"))
        trace = json.loads((self.output / "qwen_stacking.raw.json").read_text())[0]
        self.assertEqual(len(trace["requests"]), 1)
        self.assertEqual(len(trace["errors"]), 1)

    def test_qwen_boxes_replace_discovery_and_dino_but_keep_sam_and_planning(self):
        self.use_qwen_boxes({"objects": [
            {"label": "red_cube", "bbox_2d": [100, 100, 300, 300]},
            {"label": "red_cube", "bbox_2d": [400, 100, 600, 300]},
            {"label": "white_part", "bbox_2d": [700, 600, 900, 800]},
            {"label": "orange", "bbox_2d": [100, 700, 300, 900]},
        ]})
        self.expected_clean_count = 4
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--box-detector", "qwen",
                                           "--debug-visualizations"), 0)
        self.assertEqual(self.events, ["extraction", "extraction", "qwen_boxes",
                                       "segmentation", "planning", "planning"])
        self.detector_backend.assert_not_called()
        self.assertEqual(self.queries, [])
        graph = json.loads((self.output / "scene_graphs" / "scene_01.json").read_text())
        self.assertEqual([o["id"] for o in graph["objects"]], ["red_cube_1", "red_cube_2", "white_part_1", "orange_1"])
        self.assertEqual([o["label"] for o in graph["objects"]], ["red cube", "red cube", "white part", "orange"])
        self.assertEqual(graph["objects"][3]["kind"], "movable")
        self.assertIsNone(graph["objects"][3]["color"])
        self.assertTrue(all(o["confidence"] is None for o in graph["objects"]))
        raw = json.loads((self.output / "grounding_detections.json").read_text())[0]
        self.assertEqual(raw["backend"], "qwen")
        self.assertEqual(raw["detections"][0]["box"], [12, 8, 36, 24])
        segmentation = json.loads((self.output / "segmentation_results.json").read_text())[0]
        self.assertTrue(all(o["detector_score"] is None and "dino_score" not in o
                            for o in segmentation["objects"]))
        results = json.loads(self.out.read_text())["results"]
        self.assertEqual(results[0]["target_pixel"], segmentation["objects"][0]["safe_pixel"])
        trace = json.loads((self.output / "qwen_boxes.raw.json").read_text())[0]
        self.assertEqual(len(trace["requests"]), 1)
        self.assertEqual(trace["errors"], [])
        self.assertNotIn("data:image", json.dumps(trace))
        self.assertTrue((self.output / "grounding_debug" / "scene_01.png").is_file())
        discovery = json.loads((self.output / "scene_discovery.json").read_text())[0]
        self.assertNotIn("white part", discovery["instruction_objects"])
        self.assertIn("white part", discovery["qwen_objects"])
        planning_prompt = next(body["messages"][0]["content"] for url, body in self.requests
                               if "apertus.test" in url and isinstance(body["messages"][0]["content"], str)
                               and 'Instruction: "pick the red cube"' in body["messages"][0]["content"])
        self.assertIn("null when the detector supplies none", planning_prompt)

    def test_qwen_empty_inventory_builds_empty_graph_without_dino(self):
        self.use_qwen_boxes({"objects": []})
        self.expected_clean_count = 0
        self.assertEqual(self.run_pipeline("--scene", "scene_01", "--box-detector", "qwen",
                                           "--build-only"), 0)
        self.detector_backend.assert_not_called()
        graph = json.loads((self.output / "scene_graphs" / "scene_01.json").read_text())
        self.assertEqual(graph["objects"], [])
        self.assertEqual(self.events, ["extraction", "extraction", "qwen_boxes", "segmentation"])

    def test_invalid_qwen_boxes_stop_and_keep_failed_response(self):
        self.use_qwen_boxes({"objects": [123]})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_pipeline("--scene", "scene_01", "--box-detector", "qwen", "--qwen-box-retries", "0")
        self.detector_backend.assert_not_called()
        self.assertNotIn("segmentation", self.events)
        self.assertNotIn("planning", self.events)
        failure = json.loads((self.output / "perception.failure.json").read_text())
        self.assertEqual((failure["stage"], failure["scene"]), ("qwen_boxes", "scene_01"))
        trace = json.loads((self.output / "qwen_boxes.raw.json").read_text())[0]
        self.assertEqual(len(trace["requests"]), 1)
        self.assertEqual(len(trace["errors"]), 1)
        self.assertEqual(json.loads(trace["replies"][0]), {"objects": [123]})
        self.assertFalse(self.out.exists())

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
        self.assertEqual(args.stacking_source, "qwen")
        self.assertEqual(args.qwen_stacking_retries, 1)
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
