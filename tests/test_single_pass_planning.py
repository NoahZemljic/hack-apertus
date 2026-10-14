"""Verify model-owned decisions and the original format-retry budget."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import urllib.error

from src.apertus import ApertusPlanner, PlanningGenerationError
from src.validation import validate_plan


def action_plan(*steps):
    return {"decision": "plan", "steps": [{"action": action, "object_id": oid} for action, oid in steps],
            "question": None, "reason": None}


class SinglePassPlanningTests(unittest.TestCase):
    def setUp(self):
        self.graph = {"objects": [
            {"id": "bolt", "label": "bolt", "kind": "movable", "pick_point_2d": [10, 10],
             "relations": {"supports": ["nut"], "on_top_of": None}},
            {"id": "nut", "label": "nut", "kind": "movable", "pick_point_2d": [20, 20],
             "relations": {"supports": [], "on_top_of": "bolt"}},
            {"id": "tray", "label": "tray", "kind": "container", "pick_point_2d": [30, 30]},
        ], "relations": []}
        self.valid = action_plan(("pick", "bolt"))

    def test_graph_support_estimates_do_not_trigger_a_second_decision(self):
        planner = ApertusPlanner()
        before = deepcopy(self.graph)
        with patch.object(planner, "generate_response", return_value=json.dumps(self.valid)) as server:
            plan, replies = planner.plan(self.graph, "pick the bolt")
        self.assertEqual(plan, self.valid)
        self.assertEqual(self.graph, before)
        self.assertEqual(len(replies), 1)
        server.assert_called_once()
        self.assertIsNone(planner.last_trace[0]["validation_error"])

    def test_instruction_matching_and_hazards_are_sent_to_the_model(self):
        decision = {"decision": "ask_user", "steps": [], "question": "Which object?", "reason": None}
        for instruction in ("pick the blue cube", "pick it", "throw the bolt", "pick the nut"):
            with self.subTest(instruction=instruction):
                planner = ApertusPlanner()
                with patch.object(planner, "generate_response", return_value=json.dumps(decision)) as server:
                    plan, _ = planner.create_plan(self.graph, instruction)
                self.assertEqual(plan, decision)
                server.assert_called_once()
                self.assertEqual(planner.last_trace[0]["model"], planner.model)
                prompt = server.call_args.args[0][0]["content"]
                self.assertIn("Instruction: " + json.dumps(instruction), prompt)
                self.assertNotIn("Deterministic reference check", prompt)

    def test_model_refusal_is_returned_without_review_or_retry(self):
        refusal = {"decision": "refuse", "steps": [], "question": None, "reason": "The task is unsafe."}
        planner = ApertusPlanner()
        with patch.object(planner, "generate_response", return_value=json.dumps(refusal)) as server:
            actual, _ = planner.create_plan(self.graph, "throw the bolt")
        self.assertEqual(actual, refusal)
        server.assert_called_once()

    def test_original_schema_accepts_an_explanation_without_changing_the_plan(self):
        plan = dict(self.valid, reason="The bolt can be picked.")
        self.assertIs(validate_plan(plan), plan)
        planner = ApertusPlanner()
        with patch.object(planner, "generate_response", return_value=json.dumps(plan)) as server:
            actual, _ = planner.create_plan(self.graph, "pick the bolt")
        self.assertEqual(actual, plan)
        server.assert_called_once()

    def test_format_repair_preserves_the_first_request_and_has_one_retry(self):
        planner = ApertusPlanner()
        self.assertEqual(planner.retries, 1)
        with patch.object(planner, "generate_response", side_effect=["invalid JSON", json.dumps(self.valid)]) as server:
            plan, replies = planner.plan(self.graph, "pick the bolt")
        self.assertEqual(plan, self.valid)
        self.assertEqual(server.call_count, 2)
        self.assertEqual(len(replies), 2)
        self.assertEqual([trace["attempt"] for trace in planner.last_trace], [1, 2])
        self.assertEqual([trace["stage"] for trace in planner.last_trace], ["draft", "draft"])
        self.assertEqual(len(planner.last_trace[0]["messages"]), 1)
        feedback = planner.last_trace[1]["messages"][-1]["content"]
        self.assertTrue(feedback.startswith("Correct the JSON format:"))
        self.assertNotIn("Reconsider", feedback)
        self.assertNotIn("return refuse", feedback)
        self.assertIsNone(planner.last_trace[1]["validation_error"])

    def test_invalid_schema_exhausts_only_the_original_retry_budget(self):
        planner = ApertusPlanner()
        with patch.object(planner, "generate_response", return_value='{"decision":"plan"}') as server:
            with self.assertRaises(PlanningGenerationError) as error:
                planner.create_plan(self.graph, "pick the bolt")
        self.assertEqual(server.call_count, 2)
        self.assertEqual(len(error.exception.replies), 2)
        self.assertTrue(all(trace["validation_error"] for trace in planner.last_trace))

    def test_retry_override_and_trace_reset_still_work(self):
        planner = ApertusPlanner(retries=0)
        with patch.object(planner, "generate_response", return_value="invalid JSON") as server:
            with self.assertRaises(PlanningGenerationError):
                planner.create_plan(self.graph, "pick the bolt")
        server.assert_called_once()
        with patch.object(planner, "generate_response", return_value=json.dumps(self.valid)):
            planner.create_plan(self.graph, "pick the bolt")
            planner.create_plan(self.graph, "pick it again")
        self.assertEqual(len(planner.last_trace), 1)
        self.assertIn("pick it again", planner.last_trace[0]["messages"][0]["content"])

    def test_existing_schema_transport_fallback_and_errors_are_logged(self):
        planner = ApertusPlanner()
        failure = urllib.error.HTTPError("http://apertus.test", 400, "schema unsupported", {}, None)
        with patch.object(planner, "generate_response", side_effect=[failure, json.dumps(self.valid)]) as server:
            _, replies = planner.create_plan(self.graph, "pick the bolt")
        self.assertEqual(len(replies), 1)
        self.assertEqual(server.call_args.args[1], False)
        self.assertEqual(len(planner.last_trace), 2)
        self.assertIn("HTTPError", planner.last_trace[0]["request_error"])
        self.assertTrue(planner.last_trace[1]["schema_fallback"])
        with patch.object(planner, "generate_response", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(urllib.error.URLError):
                planner.create_plan(self.graph, "pick the bolt")
        self.assertEqual(len(planner.last_trace), 1)
        self.assertIn("offline", planner.last_trace[0]["request_error"])


if __name__ == "__main__":
    unittest.main()
