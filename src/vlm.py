"""Qwen model loading and inference; no scene understanding logic."""
from typing import Any, Callable

class QwenVLM:
    NORMALIZED_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        attention: str = "sdpa",
        max_new_tokens: int = 4096,
    ) -> None:
        """Configure Qwen inference without loading model weights yet"""
        self.model_name = model_name
        self.attention = attention
        self.max_new_tokens = max_new_tokens
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

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
        if self.model is not None:
            return
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as error:
            raise RuntimeError(
                "Qwen dependencies are missing. Install the vision extra on a GPU node"
            ) from error

        self.device = self.select_accelerator(torch)

        # Load once and reuse the model for every scene in the dataset
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            attn_implementation=self.attention,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.process_vision_info = process_vision_info

    def uses_normalized_coordinates(self) -> bool:
        """Qwen3.6 retains the qwen3_5 architecture name in its model config."""
        return getattr(getattr(self.model, "config", None), "model_type", "") in self.NORMALIZED_MODEL_TYPES

    def run_inference(self, messages: list[dict[str, Any]]) -> str:
        """Run inference and decode only generated tokens"""
        if any(value is None for value in (
            self.model, self.processor, self.process_vision_info
        )):
            raise RuntimeError("Qwen must be loaded before inference")

        # Format the conversation with Qwen's role markers and assistant generation prefix
        if self.uses_normalized_coordinates():
            model_inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=False,
            ).to(self.device)
        else:
            model_inputs = self.prepare_qwen25_inputs(messages)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        response_ids = [output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def prepare_qwen25_inputs(self, messages):
        """Preserve the Qwen2.5 vision preprocessing path."""
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Load and prepare the images or videos referenced by the messages
        image_inputs, video_inputs = self.process_vision_info(messages)

        # Tokenize the text, preprocess visual inputs, and move the tensors to the accelerator
        return self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
