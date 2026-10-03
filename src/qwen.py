from pathlib import Path
from typing import Any, Callable

from .json_tools import extract_json_object
from .perception import PerceptionModel
from .prompting import append_validation_feedback, prepare_perception_messages
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

    def run_inference(self, messages: list[dict[str, Any]]) -> str:
        """Run inference and decode only generated tokens"""
        if self.model is None or self.processor is None or self.process_vision_info is None:
            raise RuntimeError("Qwen must be loaded before inference")

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)

        model_inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Sampling is disabled so identical scenes produce reproducible graphs
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        # Remove input tokens before decoding the model's JSON response
        response_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
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
        if self.model is None:
            self.load_model()

        messages = prepare_perception_messages(image_path, width, height)
        responses = []
        last_error: ValueError | None = None

        for attempt in range(self.validation_retries + 1):
            response = self.run_inference(messages)
            responses.append(response)

            # Invalid model output never crosses the perception boundary
            try:
                graph = extract_json_object(response)
                return validate_scene_graph(graph, width, height), responses
            except ValueError as error:
                last_error = error

                if attempt < self.validation_retries:
                    append_validation_feedback(messages, response, error)

        raise ValueError(f"Qwen did not return a valid scene graph: {last_error}")
