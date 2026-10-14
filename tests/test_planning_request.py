"""Verify that the actual Apertus request uses the updated planning prompt."""

import io
import json
import unittest
from unittest.mock import patch

from src.apertus import ApertusPlanner
from src.prompting import PLANNING_PROMPT_PATH


class PlanningRequestTests(unittest.TestCase):
    def test_request_uses_policy_graph_and_gripper_constraint(self):
        graph = {"objects": [{"id": "red_gear_1", "label": "red_gear", "target_pixel": [10, 20],
                              "width_mm": 30, "graspable": True}], "relations": []}
        reply = {"decision": "plan", "steps": [{"action": "pick", "object_id": "red_gear_1"}],
                 "question": None, "reason": None}
        for constrained in (True, False):
            with self.subTest(constrained=constrained):
                response = io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode())
                with patch("urllib.request.urlopen", return_value=response) as http:
                    plan, replies = ApertusPlanner(constrained=constrained).create_plan(graph, "pick the red gear", 60)
                self.assertEqual(plan, reply)
                self.assertEqual(len(replies), 1)
                request = http.call_args.args[0]
                body = json.loads(request.data)
                prompt = body["messages"][0]["content"]
                self.assertTrue(prompt.startswith(PLANNING_PROMPT_PATH.read_text().strip()))
                self.assertIn(json.dumps(graph, separators=(",", ":")), prompt)
                self.assertIn("opens at most 60 mm", prompt)
                self.assertEqual("response_format" in body, constrained)


if __name__ == "__main__":
    unittest.main()
