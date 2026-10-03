import json
import urllib.error
import urllib.request

from .json_tools import extract_json_object
from .prompting import append_validation_feedback, prepare_planning_prompt
from .schemas import load_plan_schema
from .validation import validate_plan

class ApertusPlanner:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "swiss-ai/Apertus-v1.5-8B",
        timeout: float = 180,
        retries: int = 1,
        constrained: bool = True,
    ) -> None:
        """Configure access to an OpenAI-compatible Apertus server"""
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.constrained = constrained

    def request_plan(self, messages: list[dict], constrained: bool) -> str:
        """Send one planning request to Apertus and return its text response"""
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": messages
            }

        if constrained:
            # vLLM uses this schema to restrict token generation to valid JSON
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "robot_plan", "strict": True, "schema": load_plan_schema()},
            }

        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]

    def plan(self, graph: dict, instruction: str, gripper_mm: float | None) -> tuple[dict, list[str]]:
        """Generate and validate one plan, clarification question, or refusal."""
        messages = [{
            "role": "user",
            "content": prepare_planning_prompt(graph, instruction, gripper_mm),
        }]
        replies = []
        last_error = None
        constrained = self.constrained

        for attempt in range(self.retries + 1):
            try:
                reply = self.request_plan(messages, constrained)
            except urllib.error.HTTPError as error:
                # Some OpenAI-compatible servers do not implement response_format
                if constrained and error.code == 400:
                    constrained = False
                    reply = self.request_plan(messages, False)
                else:
                    raise

            replies.append(reply)

            try:
                return validate_plan(extract_json_object(reply), graph), replies
            except ValueError as error:
                last_error = error

                # Give the model one chance to correct a structurally invalid response
                if attempt < self.retries:
                    append_validation_feedback(messages, reply, error)

        raise ValueError(f"Apertus did not return a valid plan: {last_error}")
