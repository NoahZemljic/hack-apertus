import json
import urllib.error
import urllib.request
from collections.abc import Callable
from copy import deepcopy

from .json_tools import parse_json_object
from .prompting import prepare_planning_prompt
from .schemas import load_plan_response_schema
from .validation import validate_plan


class PlanningGenerationError(ValueError):
    def __init__(self, message: str, replies: list[str]) -> None:
        super().__init__(message)
        self.replies = replies


class ApertusPlanner:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "swiss-ai/Apertus-v1.5-70B",
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
        self.last_trace: list[dict] = []
        self._object_ids: list[str] | None = None

    def generate_response(self, messages: list[dict], constrained_decoding: bool) -> str:
        """Send one planning request to Apertus and return its text response"""
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2048,
            "messages": messages
            }

        if constrained_decoding:
            schema = load_plan_response_schema(self._object_ids)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "robot_plan", "strict": True,
                                "schema": schema},
            }

        # Create the inference request
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]

    def create_plan(self, graph: dict, instruction: str, gripper_mm: float | None = None) -> tuple[dict, list[str]]:
        """Ask Apertus and repair malformed output or invalid mechanical actions."""
        self.last_trace = []
        self._object_ids = [obj["id"] for obj in graph["objects"]]
        prompt = self.prompt_builder(graph, instruction, gripper_mm)
        prompt += (
            "\n\nBefore choosing decision, write assessment: one short sentence checking whether the source "
            "reference is unique after applying all qualifiers, and whether the exact requested operation "
            "violates a constraint. Then choose the matching decision and its fields. assessment is the "
            "only additional output field; do not put an explanation in reason for a plan."
            " For a transfer, emit pick(source_ID), place(destination_ID), then stop. "
            "For example, moving a washer into a tray is "
            '[{"action":"pick","object_id":"washer_ID"},'
            '{"action":"place","object_id":"tray_ID"}]. '
            "Use the actual supplied IDs, never these example IDs."
        )
        messages = [{
            "role": "user",
            "content": prompt,
        }]
        replies = []
        last_error = None
        constrained_decoding = self.constrained

        for attempt in range(self.retries + 1):
            try:
                reply, trace = self._request_with_trace(messages, constrained_decoding, "draft", attempt + 1)
            except urllib.error.HTTPError as error:
                if constrained_decoding and error.code == 400:
                    constrained_decoding = False
                    reply, trace = self._request_with_trace(messages, False, "draft", attempt + 1, fallback=True)
                else:
                    raise

            replies.append(reply)

            try:
                parsed = parse_json_object(reply)
                assessment = parsed.pop("assessment", None)
                if assessment is not None:
                    trace["assessment"] = assessment
                plan = validate_plan(parsed, graph)
                trace["validation_error"] = None
                return plan, replies
            except ValueError as error:
                last_error = error
                trace["validation_error"] = str(error)

                if attempt < self.retries:
                    messages.extend([
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": f"Correct the JSON format: {error}. "
                         "Return the full JSON object. Use only existing IDs; a place step names the destination. "
                         "Keep one object in the gripper and clear observed upper blockers before lower objects."},
                    ])

        raise PlanningGenerationError(f"Apertus did not return a valid plan: {last_error}", replies)

    def plan(self, graph: dict, instruction: str, gripper_mm: float | None = None) -> tuple[dict, list[str]]:
        """Compatibility entry point used by the grounded pipeline."""
        return self.create_plan(graph, instruction, gripper_mm)

    def _request_with_trace(self, messages: list[dict], constrained: bool, stage: str,
                            attempt: int, *, fallback: bool = False) -> tuple[str, dict]:
        """Record each actual request before inference or feedback changes messages."""
        trace = {"stage": stage, "attempt": attempt, "messages": deepcopy(messages),
                 "constrained_decoding": constrained, "model": self.model,
                 "temperature": 0, "max_tokens": 2048}
        if constrained:
            trace["response_schema"] = load_plan_response_schema(self._object_ids)
        if fallback:
            trace["schema_fallback"] = True
        self.last_trace.append(trace)
        try:
            reply = self.generate_response(messages, constrained)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            trace["request_error"] = f"{type(error).__name__}: {error}"
            raise
        trace["reply"] = reply
        return reply, trace
