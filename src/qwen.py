import json
from pathlib import Path
from typing import Any, Callable

from .decoding import SchemaConstrainedDecoder
from .perception import PerceptionGenerationError, PerceptionModel
from .prompting import PERCEPTION_RELATION_RULES, append_validation_feedback, prepare_perception_messages
from .scene_graph import derive_support_relations
from .schemas import load_perception_schema
from .validation import validate_scene_graph

class QwenSceneGraphGenerator(PerceptionModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        attention: str = "sdpa",
        max_new_tokens: int = 1800,
        validation_retries: int = 1,
    ) -> None:
        """Configure Qwen inference without loading model weights yet"""
        self.model_name = model_name
        self.attention = attention
        self.max_new_tokens = max_new_tokens
        self.validation_retries = validation_retries

        self.model: Any | None = None
        self.processor: Any | None = None
        self.process_vision_info: Callable | None = None
        self.device: str | None = None
        self.decoder: SchemaConstrainedDecoder | None = None

    @staticmethod
    def select_accelerator(torch: Any) -> str:
        """Select CUDA on CSCS or MPS for local inference"""
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        raise RuntimeError("Model inference requires a CUDA or MPS accelerator")

    def load_model(self) -> None:
        """Load optional Qwen dependencies and model weights onto an accelerator"""
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen dependencies are missing. Install the vision extra on a GPU node"
            ) from error

        self.device = self.select_accelerator(torch)

        # Load once and reuse the model for every scene in the dataset
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attention,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.process_vision_info = process_vision_info
        self.decoder = SchemaConstrainedDecoder(self.processor.tokenizer)

    def run_inference(self, messages: list[dict[str, Any]]) -> str:
        """Run inference and decode only generated tokens"""
        if any(value is None for value in (
            self.model, self.processor, self.process_vision_info, self.decoder
        )):
            raise RuntimeError("Qwen must be loaded before inference")

        # Format the conversation with Qwen's role markers and assistant generation prefix
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Load and prepare the images or videos referenced by the messages
        image_inputs, video_inputs = self.process_vision_info(messages)

        # Tokenize the text, preprocess visual inputs, and move the tensors to the accelerator
        model_inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Create fresh schema-parser state for this attempt while reusing cached tokenizer data
        token_filter = self.decoder.build_token_filter(load_perception_schema())

        # Generate greedily while restricting each next token to the schema's allowed continuations
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=token_filter,
        )

        # Generation returns the input sequence too, so retain only newly generated tokens
        response_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Decode the response into text for JSON parsing and semantic validation by the caller
        return self.processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def generate_scene_graph(
        self,
        image_path: Path,
        width: int,
        height: int,
    ) -> tuple[dict, list[str]]:
        """Generate and validate a scene graph for one image"""

        # Load the model on first use
        if self.model is None:
            self.load_model()

        messages = prepare_perception_messages(image_path, width, height)
        responses = []
        errors = []
        last_error: ValueError | None = None

        for attempt in range(self.validation_retries + 1):
            response = self.run_inference(messages)
            responses.append(response)

            # Invalid model output never crosses the perception boundary
            try:
                # Require the complete response; never accept a nested fragment of truncated JSON
                graph = json.loads(response)
                validate_scene_graph(graph, width, height, require_supports=False)

                # Generate inverse relationships in code, never by a second model prediction
                graph = derive_support_relations(graph)
                return validate_scene_graph(graph, width, height), responses
            except ValueError as error:
                last_error = error
                if isinstance(error, json.JSONDecodeError):
                    last_error = ValueError(
                        f"Invalid or incomplete JSON at line {error.lineno}, column {error.colno}: "
                        f"{error.msg}. Return a complete JSON object within {self.max_new_tokens} tokens"
                    )
                errors.append(str(last_error))

                if attempt < self.validation_retries:
                    append_validation_feedback(messages, response, last_error, PERCEPTION_RELATION_RULES)

        raise PerceptionGenerationError(
            f"Qwen did not return a valid scene graph: {last_error}", responses, errors
        )
