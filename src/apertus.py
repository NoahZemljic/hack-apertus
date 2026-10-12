import json
import urllib.error
import urllib.request
from collections.abc import Callable

from .json_tools import parse_json_object
from .schemas import load_plan_schema
from .validation import validate_plan


class PlanningGenerationError(ValueError):
    def __init__(self, message: str, replies: list[str]) -> None:
        super().__init__(message)
        self.replies = replies


class ApertusPlanner:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "swiss-ai/Apertus-v1.5-8B",
        timeout: float = 180,
        retries: int = 1,
        constrained: bool = True,
        prompt_builder: Callable[[dict, str, float | None], str] | None = None,
    ) -> None:
        """Configure access to an OpenAI-compatible Apertus server"""
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.constrained = constrained
        self.prompt_builder = prompt_builder or prepare_planning_prompt

    def generate_response(self, messages: list[dict], constrained_decoding: bool) -> str:
        """Send one planning request to Apertus and return its text response"""
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": messages
            }

        if constrained_decoding:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "robot_plan", "strict": True, "schema": load_plan_schema()},
            }

        # Create the inference request
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]

    def create_plan(self, graph: dict, instruction: str) -> tuple[dict, list[str]]:
        """Generate and validate one plan, clarification question, or refusal."""
        messages = [{
            "role": "user",
            "content": self.prompt_builder(graph, instruction, gripper_mm),
        }]
        replies = []
        last_error = None
        constrained_decoding = self.constrained_decoding

        for attempt in range(self.retries + 1):
            try:
                reply = self.generate_response(messages, constrained_decoding)
            except urllib.error.HTTPError as error:
                if constrained_decoding and error.code == 400:
                    constrained_decoding = False
                    reply = self.generate_response(messages, constrained_decoding)
                else:
                    raise

            replies.append(reply)

            try:
                return validate_plan(parse_json_object(reply)), replies
            except ValueError as error:
                last_error = error

                # Give the model one chance to correct a structurally invalid response
                if attempt < self.retries:
                    messages.extend([
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": f"Correct the JSON format: {error}. Return the full JSON object."},
                    ])

        raise PlanningGenerationError(f"Apertus did not return a valid plan: {last_error}", replies)
