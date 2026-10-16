"""Connect Apertus hints and Qwen3-VL-32B boxes to shared SAM geometry."""

from dataclasses import asdict, dataclass, field
import gc
import json
from pathlib import Path

from .geometry import GeometryConfig, GeometryReasoner
from .language.object_extraction import ask_apertus
from .scene_discovery import GENERIC_REFERENCES, build_grounding_queries
from .schemas import load_object_extraction_schema
from .settings import QWEN_MODEL


@dataclass(frozen=True)
class GroundedPerceptionConfig:
    apertus_base_url: str = "http://localhost:8000/v1"
    apertus_model: str = "swiss-ai/Apertus-v1.5-70B"
    qwen_base_url: str = "http://127.0.0.1:8001/v1"
    qwen_model: str = field(default=QWEN_MODEL, init=False)
    timeout: float = 180
    constrained: bool = True
    qwen_box_retries: int = 1
    stacking_source: str = "qwen"
    qwen_stacking_retries: int = 1
    sam_model: str = "facebook/sam2.1-hiera-small"
    sam_device: str | None = None
    debug_visualizations: bool = False


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def enrich_stacking(scenes: list[tuple[Path, object]], input_path: Path,
                    output_dir: Path, config: GroundedPerceptionConfig,
                    *, base_dir: Path | None = None) -> Path:
    """Add visual support observations to saved geometry without changing masks.

    A separate image task sees only fixed object IDs, labels and source-pixel
    boxes. Persist an explicit empty observation as well, so geometry guesses
    cannot reappear when Qwen sees no supported stack.
    """
    scene_name = None
    output_dir = Path(output_dir).resolve()
    input_path = Path(input_path).resolve()
    try:
        from .grounding.qwen_stacking import QwenStackingDetector, stacking_candidates

        reasoner = GeometryReasoner(base_dir=base_dir or input_path.parent)
        records = reasoner.load_detections(input_path)
        by_scene = {record["scene"]: record for record in records}
        missing = [scene.name for _, scene in scenes if scene.name not in by_scene]
        if missing:
            raise ValueError(f"Geometry results missing scenes: {missing}")
        # Writing a new checkpoint changes its directory. Resolve relative mask
        # references against the old checkpoint before saving it in the new run.
        for record in records:
            for obj in record["objects"]:
                mask_path = reasoner.resolve(obj["mask_path"])
                if not mask_path.is_file():
                    # Copied Clariden artifacts may still name the old scratch
                    # mount. Reuse their accompanying masks when it is absent.
                    copied_mask = input_path.parent / "segmentation" / record["scene"] / f"{obj['id']}.png"
                    if copied_mask.is_file():
                        mask_path = copied_mask
                obj["mask_path"] = str(mask_path.resolve())
        detector = QwenStackingDetector(
            model=config.qwen_model, base_url=config.qwen_base_url,
            timeout=config.timeout, constrained=config.constrained,
            retries=config.qwen_stacking_retries,
        )
        save_json(output_dir / "stacking_config.json", {
            "source": "qwen", "input_geometry": str(input_path),
            "model": config.qwen_model, "base_url": config.qwen_base_url,
            "timeout": config.timeout, "constrained": config.constrained,
            "retries": config.qwen_stacking_retries,
        })
        observations, traces = [], []
        for scene_dir, scene in scenes:
            scene_name = scene.name
            record = by_scene[scene_name]
            width, height = scene.size
            if record.get("geometry_version") != "1.0" or record.get("image") != {"width": width, "height": height}:
                raise ValueError(f"{scene_name}: saved geometry does not match the scene image")
            image_path = (scene_dir / "rgb.png").resolve()
            candidates = stacking_candidates(record["objects"])
            print(f"Inspecting stacking: {scene_name} ({len(candidates)} candidate pairs)", flush=True)
            try:
                supports = detector.detect(image_path, record["objects"])
            finally:
                traces.append({
                    "scene": scene_name, "image_path": str(image_path),
                    "candidate_pairs": candidates, "requests": detector.last_trace,
                    "replies": detector.replies, "errors": detector.errors,
                })
                save_json(output_dir / "qwen_stacking.raw.json", traces)
            record["visual_supports"] = {"backend": "qwen", "supports": supports}
            record["image_path"] = str(image_path)
            observations.append({"scene": scene_name, "backend": "qwen",
                                 "candidate_pairs": candidates, "supports": supports})
            save_json(output_dir / "stacking_relations.json", observations)
            save_json(output_dir / "geometry_results.json", records)
        return output_dir / "geometry_results.json"
    except Exception as error:
        save_json(output_dir / "perception.failure.json", {
            "stage": "qwen_stacking", "scene": scene_name,
            "error": f"{type(error).__name__}: {error}",
        })
        raise


def generate_geometry(scenes: list[tuple[Path, object]], output_dir: Path,
                      config: GroundedPerceptionConfig,
                      geometry_config: GeometryConfig | None = None) -> Path:
    """Run each existing perception stage and save every boundary.

    Evaluation categories never enter model requests. Every instruction in a
    scene contributes attention hints. Qwen discovers individual objects and boxes
    in a single visual task.
    """
    if config.qwen_box_retries < 0:
        raise ValueError("Qwen box retries must be nonnegative")
    if config.stacking_source not in {"qwen", "geometry"}:
        raise ValueError(f"Unsupported stacking source: {config.stacking_source}")
    if config.qwen_stacking_retries < 0:
        raise ValueError("Qwen stacking retries must be nonnegative")
    output_dir = output_dir.resolve()
    effective_geometry = geometry_config or GeometryConfig()
    save_json(output_dir / "perception_config.json", {
        **asdict(config), "geometry": asdict(effective_geometry),
        "scenes": [scene.name for _, scene in scenes],
    })
    stage, scene_name = "object_extraction", None
    try:
        extractions, discoveries, raw_scenes, qwen_traces = [], [], [], []
        extraction_schema = load_object_extraction_schema()
        from .grounding.qwen_boxes import QwenBoxDetector

        qwen_detector = QwenBoxDetector(
            model=config.qwen_model, base_url=config.qwen_base_url,
            timeout=config.timeout, constrained=config.constrained,
            retries=config.qwen_box_retries,
        )
        for scene_dir, scene in scenes:
            scene_name = scene.name
            hints = set()
            stage = "object_extraction"
            for instruction_id, _, instruction in scene.instructions:
                print(f"Extracting objects: {instruction_id}", flush=True)
                extraction = ask_apertus(
                    instruction, extraction_schema, base_url=config.apertus_base_url,
                    model=config.apertus_model, timeout=config.timeout, constrained=config.constrained,
                )
                extractions.append({"scene": scene.name, "instruction_id": instruction_id,
                                    "original_instruction": instruction,
                                    "extracted_objects": extraction["objects"]})
                hints.update(obj.strip().lower() for obj in extraction["objects"]
                             if obj.strip().lower() not in GENERIC_REFERENCES)
                save_json(output_dir / "object_extractions.json", extractions)
            stage = "qwen_boxes"
            print(f"Discovering objects and boxes with Qwen: {scene.name}", flush=True)
            try:
                detections = qwen_detector.detect(scene_dir / "rgb.png", sorted(hints))
            finally:
                # Retain all actual replies, including exhausted format retries.
                qwen_traces.append({
                    "scene": scene.name, "instruction_objects": sorted(hints),
                    "requests": qwen_detector.last_trace,
                    "replies": qwen_detector.replies, "errors": qwen_detector.errors,
                })
                save_json(output_dir / "qwen_boxes.raw.json", qwen_traces)
            labels = sorted({obj["label"] for obj in detections})
            raw_scenes.append({"scene": scene.name, "backend": "qwen",
                               "queries": labels, "detections": detections})
            save_json(output_dir / "grounding_detections.json", raw_scenes)
            discoveries.append({"scene": scene.name, "instruction_objects": sorted(hints),
                                "qwen_objects": labels,
                                "grounding_queries": build_grounding_queries(sorted(hints), labels)})
            save_json(output_dir / "scene_discovery.json", discoveries)

        # Saved-geometry runs do not import or load local model backends.
        stage, scene_name = "grounding", None
        from .grounding.postprocess_grounding import process_scene

        qwen_raw = {row["scene"]: row for row in raw_scenes}
        clean_scenes, reports = [], []
        scene_paths = {scene.name: directory for directory, scene in scenes}
        for discovery in discoveries:
            scene_name = discovery["scene"]
            image_path = scene_paths[scene_name] / "rgb.png"
            raw = qwen_raw[scene_name]
            detections = raw["detections"]
            stage = "grounding_cleanup"
            clean, report = process_scene(raw, image_path)
            clean_scenes.append(clean)
            reports.append(report)
            save_json(output_dir / "grounding_detections_clean.json", clean_scenes)
            save_json(output_dir / "grounding_cleanup_report.json", reports)
            if config.debug_visualizations:
                from .grounding.box_visualization import save_box_plot

                save_box_plot(image_path,
                              [("Raw QWEN", detections),
                               ("After cleanup (? = needs review)", clean["detections"])],
                              output_dir / "grounding_debug" / f"{scene_name}.png", detections)
            stage = "grounding"
        gc.collect()

        stage, scene_name = "segmentation", None
        from .masking.sam import SamSegmenter

        segmenter = SamSegmenter(config.sam_model, device=config.sam_device)
        clean_scenes = segmenter.load_detections(output_dir / "grounding_detections_clean.json")
        segmented = []
        for clean in clean_scenes:
            scene_name = clean["scene"]
            image_path = (scene_paths[scene_name] / "rgb.png").resolve()
            print(f"Segmenting {scene_name}: {len(clean['detections'])} boxes", flush=True)
            objects = segmenter.segment_scene(image_path, clean["detections"])
            for obj in objects:
                mask_path = segmenter.save_mask(obj["mask"], output_dir / "segmentation" / scene_name / f"{obj['id']}.png")
                obj["mask_path"] = str(mask_path)
            if config.debug_visualizations:
                segmenter.visualize(image_path, objects, output_dir / "segmentation_debug" / f"{scene_name}.png")
            segmented.append({"scene": scene_name, "queries": clean["queries"],
                              "image_path": str(image_path),
                              "objects": [{k: v for k, v in obj.items() if k != "mask"} for obj in objects]})
            segmenter.save_results(segmented, output_dir / "segmentation_results.json")
        del segmenter
        gc.collect()

        stage, scene_name = "geometry", None
        reasoner = GeometryReasoner(effective_geometry, base_dir=output_dir)
        records = reasoner.load_detections(output_dir / "segmentation_results.json")
        geometry, diagnostics = [], []
        for record in records:
            scene_name = record["scene"]
            print(f"Measuring geometry: {scene_name}", flush=True)
            result, debug = reasoner.analyze_scene(
                record, output_dir / "geometry_debug" if config.debug_visualizations else None,
            )
            geometry.append(result)
            diagnostics.append(debug)
            save_json(output_dir / "geometry_results.json", geometry)
            save_json(output_dir / "geometry_debug.json", {
                "config": asdict(effective_geometry), "scenes": diagnostics,
            })
        if config.stacking_source == "qwen":
            stage, scene_name = "qwen_stacking", None
            return enrich_stacking(scenes, output_dir / "geometry_results.json", output_dir, config)
        return output_dir / "geometry_results.json"
    except Exception as error:
        # Stacking records its own failing scene and exact model trace.
        if stage != "qwen_stacking":
            save_json(output_dir / "perception.failure.json", {
                "stage": stage, "scene": scene_name,
                "error": f"{type(error).__name__}: {error}",
            })
        raise
