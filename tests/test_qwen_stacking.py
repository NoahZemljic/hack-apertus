"""Exercise the separate visual support task without model weights or a server."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from PIL import Image

from src.grounding.qwen_stacking import (
    QwenStackingDetector,
    QwenStackingGenerationError,
    stacking_candidates,
)


def response(reply):
    content = reply if isinstance(reply, str) else json.dumps(reply)
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


def obj(object_id, label, box, **extra):
    return {"id": object_id, "label": label, "box": box, **extra}


class StackingCandidatesTests(unittest.TestCase):
    def test_touching_and_small_gaps_are_candidates_without_implied_direction(self):
        first = obj("red_gear_2", "red gear", [10, 10, 50, 50])
        for box in ([40, 20, 80, 60], [50, 20, 90, 60], [52, 20, 92, 60]):
            with self.subTest(box=box):
                self.assertEqual(stacking_candidates([
                    first, obj("red_gear_1", "red gear", box),
                ]), [("red_gear_2", "red_gear_1")])
        self.assertEqual(stacking_candidates([
            first, obj("red_gear_1", "red gear", [60, 20, 100, 60]),
        ]), [])

    def test_container_and_human_contact_excluded_but_mug_and_plate_support_allowed(self):
        objects = [obj("mug", "green_mug", [10, 10, 80, 80]),
                   obj("plate", "white plate", [10, 10, 80, 80])]
        for index, label in enumerate(("grey tray", "black_tray", "white container", "bin",
                                       "basket", "human hand", "left arm", "person")):
            objects.append(obj(f"excluded_{index}", label, [0, 0, 100, 100]))
        self.assertEqual(stacking_candidates(objects), [("mug", "plate")])

    def test_original_boxes_locate_contact_when_visible_masks_do_not_overlap(self):
        objects = [
            obj("upper", "cube", [20.5, 20.0, 70.0, 70.0], mask_box=[20, 20, 40, 40]),
            obj("lower", "gear", [60.0, 60.0, 100.0, 100.0], mask_box=[80, 80, 100, 100]),
        ]
        before = deepcopy(objects)
        self.assertEqual(stacking_candidates(objects), [("upper", "lower")])
        self.assertEqual(objects, before)

    def test_invalid_input_does_not_silently_hide_objects(self):
        valid = obj("gear", "gear", [10, 10, 40, 40])
        invalid = [
            [valid, deepcopy(valid)],
            [obj("gear", "", [10, 10, 40, 40])],
            [obj("gear", "gear", [10, 10, 10, 40])],
            [obj("gear", "gear", [float("nan"), 10, 40, 40])],
            [obj("gear", "gear", [False, 10, 40, 40])],
        ]
        for objects in invalid:
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                stacking_candidates(objects)


class QwenStackingDetectorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image = Path(directory.name) / "rgb.png"
        Image.new("RGB", (301, 199)).save(self.image)
        self.objects = [
            obj("red_cube_1", "red cube", [100, 100, 140, 140]),
            obj("red_gear_1", "red gear", [80, 105, 120, 145]),
            obj("red_gear_2", "red gear", [130, 105, 170, 145]),
            obj("grey_tray_1", "grey tray", [0, 0, 250, 180]),
            obj("hand_1", "hand", [90, 90, 170, 150]),
        ]

    def detector(self, **kwargs):
        return QwenStackingDetector(model="test-qwen", base_url="http://qwen.test/v1/",
                                    timeout=42, **kwargs)

    def test_one_request_preserves_inventory_and_supports_multiple_lowers(self):
        detector = self.detector()
        reply = {"supports": [
            {"upper": "red_cube_1", "lower": "red_gear_2"},
            {"upper": "red_cube_1", "lower": "red_gear_1"},
        ]}
        before = deepcopy(self.objects)
        with patch("urllib.request.urlopen", return_value=response(reply)) as http:
            self.assertEqual(detector.detect(self.image, self.objects), reply["supports"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(self.objects, before)
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, "http://qwen.test/v1/chat/completions")
        self.assertEqual(http.call_args.kwargs["timeout"], 42)
        body = json.loads(request.data)
        self.assertEqual((body["model"], body["temperature"], body["max_tokens"]),
                         ("test-qwen", 0, 4096))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["name"], "qwen_stacking")
        # XGrammar rejects array uniqueness constraints before generation.
        # Duplicate support pairs remain checked by the Python adapter.
        self.assertNotIn("uniqueItems", json.dumps(body["response_format"]["json_schema"]["schema"]))
        self.assertEqual(body["messages"][1]["content"][0]["type"], "image_url")
        prompt = body["messages"][1]["content"][1]["text"]
        self.assertIn("301 by 199", prompt)
        self.assertIn("ORIGINAL FULL-IMAGE PIXELS", prompt)
        self.assertIn('"bbox_2d": [100, 100, 140, 140]', prompt)
        self.assertNotIn('"grey_tray_1"', prompt)
        self.assertNotIn('"hand_1"', prompt)
        trace = detector.last_trace[0]
        self.assertEqual(trace["candidate_pairs"], [
            ("red_cube_1", "red_gear_1"), ("red_cube_1", "red_gear_2"),
        ])
        self.assertEqual(trace["image_size"], {"width": 301, "height": 199})
        self.assertIsNone(trace["validation_error"])
        self.assertNotIn("data:image", json.dumps(detector.last_trace))

    def test_empty_supports_is_successful_without_semantic_retry(self):
        detector = self.detector()
        with patch("urllib.request.urlopen", return_value=response({"supports": []})) as http:
            self.assertEqual(detector.detect(self.image, self.objects), [])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(detector.errors, [])

    def test_no_candidates_makes_no_visual_call_and_resets_previous_trace(self):
        detector = self.detector()
        with patch("urllib.request.urlopen", return_value=response({"supports": []})) as http:
            detector.detect(self.image, self.objects)
            self.assertEqual(detector.detect(self.image, [self.objects[0]]), [])
        self.assertEqual(http.call_count, 1)
        self.assertEqual((detector.last_trace, detector.replies, detector.errors), ([], [], []))

    def test_one_format_retry_replaces_complete_supports_answer(self):
        detector = self.detector()
        good = {"supports": [{"upper": "red_cube_1", "lower": "red_gear_1"}]}
        with patch("urllib.request.urlopen", side_effect=[response("not JSON"), response(good)]) as http:
            self.assertEqual(detector.detect(self.image, self.objects), good["supports"])
        self.assertEqual(http.call_count, 2)
        retry = json.loads(http.call_args_list[1].args[0].data)
        self.assertIn("Correct the JSON or relationship-contract error",
                      retry["messages"][1]["content"][1]["text"])
        self.assertEqual(detector.replies, ["not JSON", json.dumps(good)])
        self.assertEqual(len(detector.errors), 1)
        self.assertEqual([trace["attempt"] for trace in detector.last_trace], [1, 2])

    def test_invalid_reference_contracts_exhaust_retry_instead_of_dropping_edges(self):
        all_touching = [obj(name, "gear", [10, 10, 100, 100]) for name in ("a", "b", "c")]
        invalid = [
            ({"supports": [{"upper": "red_cube_1", "lower": "unknown"}]}, self.objects),
            ({"supports": [{"upper": "red_cube_1", "lower": "red_cube_1"}]}, self.objects),
            ({"supports": [{"upper": "red_gear_1", "lower": "red_gear_2"}]}, self.objects),
            ({"supports": [{"upper": "red_cube_1", "lower": "grey_tray_1"}]}, self.objects),
            ({"supports": [{"upper": "a", "lower": "b"},
                           {"upper": "a", "lower": "b"}]}, all_touching),
            ({"supports": [{"upper": "a", "lower": "b"},
                           {"upper": "b", "lower": "a"}]}, all_touching),
            ({"supports": [{"upper": "a", "lower": "b"},
                           {"upper": "b", "lower": "c"},
                           {"upper": "c", "lower": "a"}]}, all_touching),
            ({"supports": [{"upper": "red_cube_1", "lower": "red_gear_1", "confidence": 0.9}]},
             self.objects),
        ]
        for reply, objects in invalid:
            detector = self.detector()
            with self.subTest(reply=reply), patch("urllib.request.urlopen",
                                                side_effect=[response(reply), response(reply)]) as http:
                with self.assertRaises(QwenStackingGenerationError) as raised:
                    detector.detect(self.image, objects)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(raised.exception.replies, [json.dumps(reply), json.dumps(reply)])
            self.assertEqual(len(raised.exception.errors), 2)
            self.assertTrue(all(trace["validation_error"] for trace in detector.last_trace))

    def test_three_object_direct_chain_is_valid(self):
        objects = [obj(name, "gear", [10, 10, 100, 100]) for name in ("top", "middle", "bottom")]
        reply = {"supports": [{"upper": "top", "lower": "middle"},
                              {"upper": "middle", "lower": "bottom"}]}
        with patch("urllib.request.urlopen", return_value=response(reply)):
            self.assertEqual(self.detector().detect(self.image, objects), reply["supports"])

    def test_duplicate_json_keys_cannot_bypass_schema(self):
        detector = self.detector(retries=0)
        with patch("urllib.request.urlopen", return_value=response('{"supports": [], "supports": []}')):
            with self.assertRaises(QwenStackingGenerationError):
                detector.detect(self.image, self.objects)
        self.assertEqual(len(detector.errors), 1)

    def test_unconstrained_request_keeps_the_same_contract(self):
        with patch("urllib.request.urlopen", return_value=response({"supports": []})) as http:
            self.detector(constrained=False).detect(self.image, self.objects)
        body = json.loads(http.call_args.args[0].data)
        self.assertNotIn("response_format", body)
        self.assertIn('"supports"', body["messages"][1]["content"][1]["text"])
        self.assertEqual(http.call_count, 1)

    def test_transport_failure_is_traced_and_does_not_make_a_visual_retry(self):
        for error in (urllib.error.URLError("server unavailable"), TimeoutError("timed out"),
                      urllib.error.HTTPError("http://qwen.test", 400, "bad request", {}, None)):
            detector = self.detector(retries=4)
            with self.subTest(error=error), patch("urllib.request.urlopen", side_effect=error) as http:
                with self.assertRaises(type(error)):
                    detector.detect(self.image, self.objects)
            self.assertEqual(http.call_count, 1)
            self.assertEqual((detector.replies, detector.errors), ([], []))
            self.assertIn("request_error", detector.last_trace[0])

    def test_invalid_server_envelopes_are_traced_without_model_format_retry(self):
        for envelope in (b"not JSON", b'{"choices": []}', b'{}'):
            detector = self.detector()
            with self.subTest(envelope=envelope), patch("urllib.request.urlopen",
                                                      return_value=io.BytesIO(envelope)) as http:
                with self.assertRaises(ValueError):
                    detector.detect(self.image, self.objects)
            self.assertEqual(http.call_count, 1)
            self.assertEqual((detector.replies, detector.errors), ([], []))
            self.assertIn("request_error", detector.last_trace[0])

    def test_http_rejection_preserves_server_reason_without_another_visual_call(self):
        detector = self.detector()
        body = json.dumps({"error": {"message": "The provided JSON schema contains features not supported by xgrammar."}})
        error = urllib.error.HTTPError("http://qwen.test", 400, "Bad Request", {}, io.BytesIO(body.encode()))
        with patch("urllib.request.urlopen", side_effect=error) as http:
            with self.assertRaisesRegex(urllib.error.HTTPError, "not supported by xgrammar"):
                detector.detect(self.image, self.objects)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(detector.last_trace[0]["http_status"], 400)
        self.assertEqual(detector.last_trace[0]["http_error_body"], body)
        self.assertEqual((detector.replies, detector.errors), ([], []))

    def test_out_of_image_input_is_not_sent_to_qwen(self):
        objects = [obj("a", "gear", [10, 10, 400, 100]), obj("b", "gear", [10, 10, 100, 100])]
        with patch("urllib.request.urlopen") as http:
            with self.assertRaisesRegex(ValueError, "within the source image"):
                self.detector().detect(self.image, objects)
        http.assert_not_called()

    def test_retries_must_be_nonnegative_integers(self):
        for retries in (-1, 1.5, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                self.detector(retries=retries)


if __name__ == "__main__":
    unittest.main()
