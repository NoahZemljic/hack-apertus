"""Exercise Qwen's HTTP detector boundary without model weights or a server."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.grounding.qwen_boxes import QwenBoxDetector, QwenBoxGenerationError


def response(reply):
    content = reply if isinstance(reply, str) else json.dumps(reply)
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


class QwenBoxDetectorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image = Path(directory.name) / "rgb.png"
        Image.new("RGB", (301, 199)).save(self.image)

    def detector(self, **kwargs):
        return QwenBoxDetector(model="test-qwen", base_url="http://qwen.test/v1/",
                               timeout=42, **kwargs)

    def test_joint_request_keeps_repeated_instances_and_objects_beyond_hints(self):
        reply = {"objects": [
            {"label": "red_gear", "bbox_2d": [100, 200, 300, 400]},
            {"label": "red_gear", "bbox_2d": [600, 700, 900, 900]},
            {"label": "Green_Mug", "bbox_2d": [500, 100, 700, 300]},
            {"label": "mandarin", "bbox_2d": [800, 100, 900, 200]},
        ]}
        detector = self.detector()
        with patch("urllib.request.urlopen", return_value=response(reply)) as http:
            detections = detector.detect(self.image, ["red gear", "invisible blue cube"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual([row["label"] for row in detections], ["red gear", "red gear", "green mug", "mandarin"])
        self.assertEqual([row["id"] for row in detections], ["red_gear_1", "red_gear_2", "green_mug_1", "mandarin_1"])
        self.assertTrue(all(row["score"] is None and row["backend"] == "qwen" for row in detections))
        self.assertTrue(all(row["query"] == row["label"] for row in detections))
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, "http://qwen.test/v1/chat/completions")
        self.assertEqual(http.call_args.kwargs["timeout"], 42)
        body = json.loads(request.data)
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"]), ("test-qwen", 4096, 0))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        prompt = body["messages"][1]["content"][1]["text"]
        self.assertIn('"invisible blue cube"', prompt)
        self.assertIn("including objects outside the attention hints", prompt)
        self.assertEqual(body["messages"][1]["content"][0]["type"], "image_url")
        self.assertIn("black tray", body["messages"][0]["content"])
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(detector.replies, [json.dumps(reply)])
        self.assertIsNone(detector.last_trace[0]["validation_error"])
        self.assertEqual(detector.last_trace[0]["chat_template_kwargs"], body["chat_template_kwargs"])
        self.assertNotIn("data:image", json.dumps(detector.last_trace))

    def test_nonsquare_source_scaling_uses_floor_and_ceil_and_keeps_edges(self):
        reply = {"objects": [
            {"label": "grey tray", "bbox_2d": [0, 0, 1000, 1000]},
            {"label": "white component", "bbox_2d": [101, 203, 304, 407]},
        ]}
        with patch("urllib.request.urlopen", return_value=response(reply)):
            detections = self.detector().detect(self.image, [])
        self.assertEqual(detections[0]["box"], [0, 0, 301, 199])
        self.assertEqual(detections[1]["box"], [30, 40, 92, 81])

    def test_equivalent_spelling_and_plural_labels_share_instance_ids(self):
        reply = {"objects": [
            {"label": "gray gears", "bbox_2d": [100, 100, 200, 200]},
            {"label": "grey gear", "bbox_2d": [400, 400, 500, 500]},
        ]}
        with patch("urllib.request.urlopen", return_value=response(reply)):
            detections = self.detector().detect(self.image, [])
        self.assertEqual([d["label"] for d in detections], ["grey gear", "grey gear"])
        self.assertEqual([d["id"] for d in detections], ["grey_gear_1", "grey_gear_2"])

    def test_valid_empty_inventory_succeeds_without_second_visual_pass(self):
        detector = self.detector()
        with patch("urllib.request.urlopen", return_value=response({"objects": []})) as http:
            self.assertEqual(detector.detect(self.image, ["red gear"]), [])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(detector.errors, [])

    def test_malformed_output_gets_one_format_retry_then_returns_full_inventory(self):
        detector = self.detector()
        reply = {"objects": [{"label": "red gear", "bbox_2d": [100, 100, 200, 200]}]}
        with patch("urllib.request.urlopen", side_effect=[response("not JSON"), response(reply)]) as http:
            detections = detector.detect(self.image, ["red gear"])
        self.assertEqual(len(detections), 1)
        self.assertEqual(http.call_count, 2)
        retry_body = json.loads(http.call_args_list[1].args[0].data)
        self.assertIn("Correct the JSON format or coordinate error", retry_body["messages"][1]["content"][1]["text"])
        self.assertEqual([trace["attempt"] for trace in detector.last_trace], [1, 2])
        self.assertEqual(detector.replies, ["not JSON", json.dumps(reply)])
        self.assertEqual(len(detector.errors), 1)

    def test_retry_exhaustion_retains_both_model_replies_and_errors(self):
        detector = self.detector()
        bad = {"objects": [{"label": "red gear", "bbox_2d": [200, 100, 100, 200]}]}
        with patch("urllib.request.urlopen", side_effect=[response("not JSON"), response(bad)]) as http:
            with self.assertRaises(QwenBoxGenerationError) as raised:
                detector.detect(self.image, [])
        self.assertEqual(http.call_count, 2)
        self.assertEqual(raised.exception.replies, ["not JSON", json.dumps(bad)])
        self.assertEqual(len(raised.exception.errors), 2)
        self.assertIn("positive-area", raised.exception.errors[-1])

    def test_schema_and_coordinate_violations_are_rejected_without_clipping(self):
        invalid_objects = [
            {"label": "red gear", "bbox_2d": [-1, 0, 100, 100]},
            {"label": "red gear", "bbox_2d": [0, 0, 1001, 100]},
            {"label": "red gear", "bbox_2d": [100, 0, 100, 100]},
            {"label": "red gear", "bbox_2d": [0, 100, 100, 0]},
            {"label": "red gear", "bbox_2d": [0, 0, 100, 0]},
            {"label": "red gear", "bbox_2d": [0, 0, 1.5, 100]},
            {"label": "red gear", "bbox_2d": [0, 0, 100.0, 100]},
            {"label": "red gear", "bbox_2d": [False, 0, 100, 100]},
            {"label": "red gear", "bbox_2d": [0, 0, 100]},
            {"label": "red gear", "bbox_2d": [0, 0, 100, 100], "confidence": 0.9},
            {"label": "red gear", "bbox_2d": [0, 0, 100, 100], "target_pixel": [10, 10]},
        ]
        for obj in invalid_objects:
            with self.subTest(obj=obj), patch("urllib.request.urlopen", return_value=response({"objects": [obj]})) as http:
                with self.assertRaises(QwenBoxGenerationError):
                    self.detector(retries=0).detect(self.image, [])
                self.assertEqual(http.call_count, 1)

    def test_duplicate_keys_and_nonstandard_numbers_are_rejected(self):
        for reply in ('{"objects": [], "objects": []}',
                      '{"objects": [{"label": "red gear", "bbox_2d": [0,0,NaN,100]}]}'):
            with self.subTest(reply=reply), patch("urllib.request.urlopen", return_value=response(reply)):
                with self.assertRaises(QwenBoxGenerationError):
                    self.detector(retries=0).detect(self.image, [])

    def test_unconstrained_request_uses_same_contract_and_one_pass(self):
        with patch("urllib.request.urlopen", return_value=response({"objects": []})) as http:
            self.detector(constrained=False).detect(self.image, [])
        body = json.loads(http.call_args.args[0].data)
        self.assertNotIn("response_format", body)
        self.assertIn("bbox_2d", body["messages"][1]["content"][1]["text"])
        self.assertEqual(http.call_count, 1)

    def test_transport_failure_is_not_retried_or_replaced_with_empty_inventory(self):
        detector = self.detector(retries=4)
        error = urllib.error.URLError("server unavailable")
        with patch("urllib.request.urlopen", side_effect=error) as http:
            with self.assertRaises(urllib.error.URLError):
                detector.detect(self.image, [])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(detector.replies, [])
        self.assertIn("server unavailable", detector.last_trace[0]["request_error"])

    def test_http_error_does_not_add_a_schema_fallback_visual_call(self):
        error = urllib.error.HTTPError("http://qwen.test", 400, "bad request", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as http:
            with self.assertRaises(urllib.error.HTTPError):
                self.detector().detect(self.image, [])
        self.assertEqual(http.call_count, 1)

    def test_malformed_server_envelope_is_logged_without_a_visual_retry(self):
        detector = self.detector()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"not an API response")) as http:
            with self.assertRaisesRegex(ValueError, "response envelope"):
                detector.detect(self.image, [])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(detector.replies, [])
        self.assertEqual(detector.errors, [])
        self.assertIn("response envelope", detector.last_trace[0]["request_error"])

    def test_trace_and_instance_ids_reset_between_scenes(self):
        detector = self.detector()
        reply = {"objects": [{"label": "red gear", "bbox_2d": [0, 0, 100, 100]}]}
        with patch("urllib.request.urlopen", side_effect=[response("bad"), response(reply), response(reply)]):
            detector.detect(self.image, [])
            second = detector.detect(self.image, [])
        self.assertEqual(second[0]["id"], "red_gear_1")
        self.assertEqual(len(detector.last_trace), 1)
        self.assertEqual(len(detector.replies), 1)
        self.assertEqual(detector.errors, [])

    def test_retries_must_be_nonnegative_integers(self):
        for retries in (-1, 1.5, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                self.detector(retries=retries)


if __name__ == "__main__":
    unittest.main()
