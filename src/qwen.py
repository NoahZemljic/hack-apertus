import json
import tempfile
import math
import time
from pathlib import Path
from typing import Any, Callable

from .decoding import SchemaConstrainedDecoder
from .json_tools import parse_json_object
from .perception import PerceptionGenerationError, PerceptionModel
from .prompting import prepare_perception_messages
from .scene_graph import derive_support_relations
from .schemas import load_perception_schema
from .validation import validate_scene_graph

class QwenSceneGraphGenerator(PerceptionModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        attention: str = "sdpa",
        max_new_tokens: int = 4096,
        validation_retries: int = 2,
    ) -> None:
        """Configure Qwen inference without loading model weights yet"""
        self.model_name = model_name
        self.attention = attention
        self.max_new_tokens = max_new_tokens
        self.validation_retries = validation_retries
        if max_new_tokens < 1 or validation_retries < 0:
            raise ValueError("max_new_tokens must be positive and validation_retries nonnegative")

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
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as error:
            raise RuntimeError(
                "Qwen dependencies are missing. Install the vision extra on a GPU node"
            ) from error

        self.device = self.select_accelerator(torch)

        # Load once and reuse the model for every scene in the dataset
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attention,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.process_vision_info = process_vision_info
        self.decoder = SchemaConstrainedDecoder(self.processor.tokenizer)

    def run_inference(self, messages: list[dict[str, Any]], schema: dict) -> str:
        """Run inference and decode only generated tokens"""
        if any(value is None for value in (
            self.model, self.processor, self.process_vision_info, self.decoder
        )):
            raise RuntimeError("Qwen must be loaded before inference")

        # Format the conversation with Qwen's role markers and assistant generation prefix
        if getattr(self.model.config, "model_type", "") == "qwen3_vl":
            model_inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
        else:
            model_inputs = self.prepare_qwen25_inputs(messages)

        # Create fresh schema-parser state for this attempt while reusing cached tokenizer data
        token_filter = self.decoder.build_token_filter(schema)
        properties = schema.get("properties", {})
        budget = self.max_new_tokens if "objects" in properties else min(self.max_new_tokens, 512)
        if set(properties) == {"point"}:
            budget = min(budget, 128)
        started = time.monotonic()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=budget,
            do_sample=False,
            prefix_allowed_tokens_fn=token_filter,
        )
        response_ids = [output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        print(f"  Qwen: {len(response_ids[0])} tokens in {time.monotonic() - started:.1f}s", flush=True)
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

        if getattr(getattr(self.model, "config", None), "model_type", "") == "qwen3_vl":
            graph, responses = self.generate_full_graph(image_path, 1000, 1000, normalized=True)
            for obj in graph["objects"]:
                x1, y1, x2, y2 = obj["bbox_2d"]
                obj["bbox_2d"] = [math.floor(x1 * width / 1000), math.floor(y1 * height / 1000),
                                  math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000)]
                u, v = obj["pick_point_2d"]
                obj["pick_point_2d"] = [math.floor(u * width / 1000), math.floor(v * height / 1000)]
            graph["image"] = {"width": width, "height": height}
            try:
                self.detect_missing_work_mats(graph, image_path, responses)
                self.refine_contact_relations(graph, image_path, responses)
                self.reconcile_contact_pairs(graph, image_path, responses)
                self.refine_normalized_points(graph, image_path, responses)
                responses.append(json.dumps(graph))
                return validate_scene_graph(derive_support_relations(graph), width, height), responses
            except PerceptionGenerationError:
                raise
            except ValueError as error:
                responses.append(json.dumps(graph))
                raise PerceptionGenerationError(f"Qwen refinement failed: {error}", responses, [str(error)]) from error

        return self.generate_full_graph(image_path, width, height)

    @staticmethod
    def parse_structured_response(response, contract):
        from jsonschema import Draft202012Validator
        result = parse_json_object(response)
        errors = list(Draft202012Validator(contract).iter_errors(result))
        if errors:
            raise ValueError("; ".join(error.message for error in errors))
        return result

    def detect_missing_work_mats(self, graph, image_path, responses):
        if any(obj["label"].endswith("mat") for obj in graph["objects"]):
            return
        contract = {"type": "object", "required": ["mats"], "additionalProperties": False,
                    "properties": {"mats": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["bbox_2d", "color", "confidence", "partially_occluded"], "properties": {
                            "bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                            "color": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "partially_occluded": {"type": "boolean"}}}}}}
        prompt = (
            "Locate visible flat work mats used as placement areas. In this robot workspace the flat "
            "black rectangular work area is called the black tray, even though it has no enclosing rim. "
            "Include it when empty. Only detect visible usable placement areas; do not infer mats beneath "
            "trays or label the whole image/table as a mat. If no mat is visible return mats: []. "
            "Return tight boxes in NORMALIZED 0..1000 coordinates, one axis spanning each full image dimension. "
            f"Return JSON matching {json.dumps(contract)}"
        )
        response = self.run_inference([{"role": "user", "content": [
            {"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": prompt},
        ]}], contract)
        responses.append(response)
        from .validation import validate_pixel
        result = self.parse_structured_response(response, contract)
        width, height = graph["image"]["width"], graph["image"]["height"]
        ids = {obj["id"] for obj in graph["objects"]}
        for mat in result["mats"]:
            x1, y1, x2, y2 = validate_pixel(mat["bbox_2d"], "work mat", 1000, 1000, 4)
            index = 1
            while f"work_mat_{index}" in ids:
                index += 1
            object_id = f"work_mat_{index}"
            ids.add(object_id)
            box = [math.floor(x1 * width / 1000), math.floor(y1 * height / 1000),
                   math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000)]
            graph["objects"].append(dict(id=object_id, kind="surface", label="work mat", color=mat["color"],
                bbox_2d=box, pick_point_2d=[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2], confidence=mat["confidence"],
                relations=dict(inside=None, on_top_of=[], supports=[], held_by_human=False,
                               partially_occluded=mat["partially_occluded"])))

    def refine_contact_relations(self, graph, image_path, responses):
        """Inspect potentially overlapping objects together without deriving contact from overlap."""
        import cv2

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        objects = graph["objects"]
        with tempfile.TemporaryDirectory(prefix="qwen-contacts-") as directory:
            for index, obj in enumerate(objects):
                if obj["kind"] not in {"movable", "surface"} or obj["label"].endswith("mat"):
                    continue
                x1, y1, x2, y2 = obj["bbox_2d"]
                neighbors = [other for other in objects if other is not obj
                             and other["kind"] != "container" and not other["label"].endswith("mat")
                             and min(x2, other["bbox_2d"][2]) > max(x1, other["bbox_2d"][0])
                             and min(y2, other["bbox_2d"][3]) > max(y1, other["bbox_2d"][1])]
                unassigned_mat_support = not obj["relations"]["on_top_of"] and any(
                    other["label"].endswith("mat")
                    and min(x2, other["bbox_2d"][2]) > max(x1, other["bbox_2d"][0])
                    and min(y2, other["bbox_2d"][3]) > max(y1, other["bbox_2d"][1])
                    for other in objects
                )
                if not neighbors and not unassigned_mat_support:
                    continue
                boxes = [obj["bbox_2d"], *[other["bbox_2d"] for other in neighbors]]
                left, top = max(0, min(box[0] for box in boxes) - 60), max(0, min(box[1] for box in boxes) - 60)
                right, bottom = min(width, max(box[2] for box in boxes) + 60), min(height, max(box[3] for box in boxes) + 60)
                crop = Path(directory) / f"contact_{index}.png"
                cv2.imwrite(str(crop), image[top:bottom, left:right])
                parents = set(obj["relations"]["on_top_of"])
                parents.update(other["id"] for other in objects if other["kind"] == "container" or other["label"].endswith("mat"))
                candidates = [other for other in objects if other is not obj and other["kind"] != "human"
                              and (other in neighbors or other["id"] in parents)]
                items = {"type": "string", "enum": [other["id"] for other in candidates]} if candidates else {"type": "string"}
                fields = {"observation": {"type": "string", "maxLength": 240},
                          "direct_supports_below_target": {"type": "array", "items": items, "uniqueItems": True},
                          "held_by_human": {"type": "boolean"}, "partially_occluded": {"type": "boolean"}}
                if not candidates:
                    fields["direct_supports_below_target"]["maxItems"] = 0
                contract = {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}
                descriptions = []
                for other in [obj, *neighbors, *[parent for parent in candidates if parent not in neighbors]]:
                    a, b, c, d = other["bbox_2d"]
                    descriptions.append({"id": other["id"], "label": other["label"], "color": other["color"],
                        "crop_box": [round((a-left)*1000/(right-left)), round((b-top)*1000/(bottom-top)),
                                     round((c-left)*1000/(right-left)), round((d-top)*1000/(bottom-top))]})
                prompt = (
                    f"Inspect ONLY target {obj['id']} ({obj['color']} {obj['label']}) in this cropped image. "
                    "Look at the boundary where the target meets each nearby object. Which object hides "
                    "part of the other and lies above it? First describe this visible boundary in one "
                    "short observation sentence, then report the support and occlusion fields. "
                    "Which objects directly UNDER the target carry its weight? Return their IDs in "
                    "direct_supports_below_target. Do NOT list objects above the target. A top gear lying "
                    "on a bottom gear is supported by the bottom gear, not by the mat further underneath. "
                    "A cube bridging two gears lists both gears. If objects are merely adjacent use only "
                    "the actual base support, or [] if none is represented. Bounding-box overlap alone "
                    "does not prove contact. Use visible occlusion and contact. "
                    "partially_occluded is true when another object hides part of the TARGET. "
                    "held_by_human is true when a hand grips the TARGET. "
                    "Inventory boxes are relative to this crop in normalized coordinates; objects extending "
                    f"outside the crop can have coordinates outside 0..1000: {json.dumps(descriptions)}\n"
                    f"Return JSON matching {json.dumps(contract)}"
                )
                response = self.run_inference([{"role": "user", "content": [
                    {"type": "image", "image": str(crop)}, {"type": "text", "text": prompt},
                ]}], contract)
                responses.append(response)
                detail = self.parse_structured_response(response, contract)
                obj["relations"].update(on_top_of=detail["direct_supports_below_target"],
                    held_by_human=detail["held_by_human"], partially_occluded=detail["partially_occluded"])

    def reconcile_contact_pairs(self, graph, image_path, responses):
        """Ask once per overlapping pair so opposite directional answers cannot coexist."""
        import cv2

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        objects = [obj for obj in graph["objects"] if obj["kind"] in {"movable", "surface"}
                   and not obj["label"].endswith("mat")]
        occluded = {obj["id"]: obj["relations"]["held_by_human"] for obj in objects}
        reviewed = set()
        bases = [obj for obj in graph["objects"] if obj["kind"] == "container" or obj["label"].endswith("mat")]
        with tempfile.TemporaryDirectory(prefix="qwen-pairs-") as directory:
            for index, a in enumerate(objects):
                for b in objects[index + 1:]:
                    ax, ay, az, aw = a["bbox_2d"]
                    bx, by, bz, bw = b["bbox_2d"]
                    intersection = max(0, min(az, bz) - max(ax, bx)) * max(0, min(aw, bw) - max(ay, by))
                    smaller_area = min((az - ax) * (aw - ay), (bz - bx) * (bw - by))
                    # Loose boxes can touch at corners despite a visible gap. Do not
                    # force a front/back question for negligible projected overlap.
                    if intersection < 0.1 * smaller_area:
                        continue
                    suspected_contact = (a["relations"]["partially_occluded"]
                                         or b["relations"]["partially_occluded"]
                                         or b["id"] in a["relations"]["on_top_of"]
                                         or a["id"] in b["relations"]["on_top_of"])
                    if not suspected_contact:
                        continue
                    left, top = max(0, min(ax, bx) - 30), max(0, min(ay, by) - 30)
                    right, bottom = min(width, max(az, bz) + 30), min(height, max(aw, bw) + 30)
                    path = Path(directory) / "pair.png"
                    cv2.imwrite(str(path), image[top:bottom, left:right])
                    contract = {"type": "object", "additionalProperties": False,
                                "required": ["front_object", "direct_contact", "lower_base_support"], "properties": {
                                    "front_object": {"enum": [a["id"], b["id"], "neither", "unclear"]},
                                    "direct_contact": {"type": "boolean"},
                                    "lower_base_support": {"enum": [obj["id"] for obj in bases] + ["none"]}}}
                    descriptions = []
                    for obj in (a, b):
                        x1, y1, x2, y2 = obj["bbox_2d"]
                        descriptions.append(dict(id=obj["id"], color=obj["color"], label=obj["label"],
                            crop_box=[round((x1-left)*1000/(right-left)), round((y1-top)*1000/(bottom-top)),
                                      round((x2-left)*1000/(right-left)), round((y2-top)*1000/(bottom-top))]))
                    prompt = (
                        f"Look at the boundary where these two objects meet: {json.dumps(descriptions)}. "
                        "Which object hides part of the other and lies above it? Return that object's ID "
                        "as front_object. Above means closer to the camera, NOT higher up in the image. "
                        "Use neither if they are side by side without occlusion, unclear if you cannot tell. "
                        "direct_contact is true only if the front object rests directly on the other object. "
                        "lower_base_support is the visible tray or mat directly carrying the lower object, "
                        "or none if absent/unclear. Available bases: "
                        f"{json.dumps([dict(id=obj['id'], label=obj['label'], color=obj['color']) for obj in bases])}. "
                        f"Return only JSON matching {json.dumps(contract)}"
                    )
                    response = self.run_inference([{"role": "user", "content": [
                        {"type": "image", "image": str(path)}, {"type": "text", "text": prompt},
                    ]}], contract)
                    responses.append(response)
                    detail = self.parse_structured_response(response, contract)
                    if detail["front_object"] == "unclear":
                        raise ValueError(f"Unresolved contact between {a['id']} and {b['id']}")
                    reviewed.update((a["id"], b["id"]))
                    for obj, other in ((a, b), (b, a)):
                        obj["relations"]["on_top_of"] = [parent for parent in obj["relations"]["on_top_of"]
                                                            if parent != other["id"]]
                    if detail["front_object"] in {a["id"], b["id"]}:
                        upper, lower = (a, b) if detail["front_object"] == a["id"] else (b, a)
                        occluded[lower["id"]] = True
                        if detail["direct_contact"]:
                            upper["relations"]["on_top_of"].append(lower["id"])
                            if not lower["relations"]["on_top_of"] and detail["lower_base_support"] != "none":
                                lower["relations"]["on_top_of"].append(detail["lower_base_support"])
            # A represented object directly beneath a target replaces the distant base.
            object_ids = {obj["id"] for obj in objects}
            for obj in objects:
                if obj["id"] in reviewed:
                    obj["relations"]["partially_occluded"] = occluded[obj["id"]]
                parents = obj["relations"]["on_top_of"]
                if any(parent in object_ids for parent in parents):
                    obj["relations"]["on_top_of"] = [parent for parent in parents if parent in object_ids]

    def refine_normalized_points(self, graph: dict, image_path: Path, responses: list[str]) -> None:
        """Ground points in isolated crops; normalize once, then translate to source pixels."""
        import cv2
        from .validation import validate_pixel

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        contract = {"type": "object", "additionalProperties": False, "required": ["point"],
                    "properties": {"point": {"type": "array", "items": {"type": "integer", "minimum": 0},
                                             "minItems": 2, "maxItems": 2}}}
        with tempfile.TemporaryDirectory(prefix="qwen-points-") as directory:
            for index, obj in enumerate(graph["objects"]):
                x1, y1, x2, y2 = obj["bbox_2d"]
                padding = max(20, round(max(x2 - x1, y2 - y1) * 0.1))
                left, top = max(0, x1 - padding), max(0, y1 - padding)
                right, bottom = min(width, x2 + padding), min(height, y2 + padding)
                path = Path(directory) / f"crop_{index}.png"
                cv2.imwrite(str(path), image[top:bottom, left:right])
                errors = []
                for attempt in range(self.validation_retries + 1):
                    prompt = (
                        f"Locate a point on the {obj['color']} {obj['label']} in this close-up. "
                        "Use NORMALIZED coordinates [x,y] from 0 to 999; the full width and height each span 0..1000. "
                        "Choose exposed solid material of the target, never a hole or another object. "
                        "For a mug choose the visible handle material, not the open cup cavity or the hole "
                        "in the handle; if the handle is covered by a hand choose visible mug rim material. "
                        "For a metal part avoid ALL bores and mounting holes. For a gear choose the visible solid hub. "
                        "For a tray, mat, or plate choose an empty surface spot away from objects. "
                        "For a hand choose a visible point on the hand. "
                        'Return only {"point": [x,y]}.'
                    )
                    if errors:
                        prompt += f" Previous invalid answer: {responses[-1]}. Error: {errors[-1]}. Reinspect."
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": str(path)}, {"type": "text", "text": prompt},
                    ]}]
                    response = self.run_inference(messages, contract)
                    responses.append(response)
                    try:
                        result = self.parse_structured_response(response, contract)
                        point = validate_pixel(result.get("point"), obj["id"], 1000, 1000, 2)
                        u = left + math.floor(point[0] * (right - left) / 1000)
                        v = top + math.floor(point[1] * (bottom - top) / 1000)
                        if not (x1 <= u < x2 and y1 <= v < y2):
                            raise ValueError(f"Point {[u, v]} outside target box {obj['bbox_2d']}; select the target itself")
                        obj["pick_point_2d"] = [u, v]
                        from .grounding import refine_material_point
                        material_check = refine_material_point(image, obj, graph["objects"])
                        if material_check is not None:
                            obj["pick_point_2d"] = material_check["selected"]
                            responses.append(json.dumps(material_check))
                        break
                    except ValueError as error:
                        errors.append(str(error))
                else:
                    raise PerceptionGenerationError(f"Point refinement failed for {obj['id']}", responses, errors)

    def generate_full_graph(self, image_path: Path, width: int, height: int,
                            normalized: bool = False) -> tuple[dict, list[str]]:
        def prepare_messages():
            messages = prepare_perception_messages(image_path, width, height)
            if normalized:
                content = messages[0]["content"][-1]
                content["text"] = content["text"].replace(
                    "The original image is 1000 x 1000 pixels.",
                    "Use a NORMALIZED 1000 x 1000 coordinate plane: x=0..1000 spans the full image width "
                    "and y=0..1000 spans the full image height. The image field describes this coordinate "
                    "plane, not the source resolution. Code converts these coordinates to source pixels.",
                ).replace("in this original image space", "in this normalized coordinate plane")
            return messages

        messages = prepare_messages()
        schema = load_perception_schema(width, height)
        responses = []
        errors = []
        last_error: ValueError | None = None

        for attempt in range(self.validation_retries + 1):
            response = self.run_inference(messages, schema)
            responses.append(response)

            # Invalid model output never crosses the perception boundary
            try:
                # Require the complete response; never accept a nested fragment of truncated JSON
                graph = parse_json_object(response)
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
                print(f"  Qwen validation attempt {attempt + 1}: {last_error}", flush=True)

                if len(errors) > self.validation_retries:
                    break
                # A fresh image-grounded request avoids growing an assistant history
                # containing an invalid answer that Qwen may simply repeat.
                messages = prepare_messages()
                messages[0]["content"].append({"type": "text", "text": (
                    "A previous candidate failed validation. Reinspect the original image and "
                    "regenerate the complete graph, correcting every listed error. "
                    "The candidate is not evidence and must not be copied unchanged.\n"
                    f"Errors:\n{last_error}\nRejected candidate:\n{response}"
                )})

        raise PerceptionGenerationError(
            f"Qwen did not return a valid scene graph: {last_error}", responses, errors
        )
