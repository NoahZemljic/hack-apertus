"""Connect the existing extraction, discovery, grounding, masking, and geometry stages."""

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path

from .geometry import GeometryConfig, GeometryReasoner
from .language.object_extraction import ask_apertus
from .scene_discovery import GENERIC_REFERENCES, build_grounding_queries, discover_scene
from .schemas import load_object_extraction_schema, load_scene_discovery_schema


@dataclass(frozen=True)
class GroundedPerceptionConfig:
    apertus_base_url: str = "http://localhost:8000/v1"
    apertus_model: str = "swiss-ai/Apertus-v1.5-70B"
    qwen_base_url: str = "http://127.0.0.1:8001/v1"
    qwen_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    timeout: float = 180
    constrained: bool = True
    query_source: str = "qwen_objects"
    dino_model: str = "IDEA-Research/grounding-dino-base"
    dino_device: str | None = None
    sam_model: str = "facebook/sam2.1-hiera-small"
    sam_device: str | None = None
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    debug_visualizations: bool = False


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def generate_geometry(scenes: list[tuple[Path, object]], output_dir: Path,
                      config: GroundedPerceptionConfig,
                      geometry_config: GeometryConfig | None = None) -> Path:
    """Run each existing perception stage and save every boundary.

    Categories never enter model requests. Discovery uses every instruction in
    a selected scene. Verified Qwen labels reach DINO by default.
    """
    if config.query_source not in {"qwen_objects", "grounding_queries"}:
        raise ValueError(f"Unsupported query source: {config.query_source}")
    output_dir = output_dir.resolve()
    effective_geometry = geometry_config or GeometryConfig()
    save_json(output_dir / "perception_config.json", {
        **asdict(config), "geometry": asdict(effective_geometry),
        "scenes": [scene.name for _, scene in scenes],
    })
    stage, scene_name = "object_extraction", None
    try:
        extractions, discoveries = [], []
        extraction_schema = load_object_extraction_schema()
        discovery_schema = load_scene_discovery_schema()
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
            stage = "scene_discovery"
            print(f"Discovering visible objects: {scene.name}", flush=True)
            labels = discover_scene(
                None, scene_dir / "rgb.png", discovery_schema, sorted(hints),
                model=config.qwen_model, base_url=config.qwen_base_url,
                timeout=config.timeout, constrained=config.constrained,
            )
            discoveries.append({"scene": scene.name, "instruction_objects": sorted(hints),
                                "qwen_objects": labels,
                                "grounding_queries": build_grounding_queries(sorted(hints), labels)})
            save_json(output_dir / "scene_discovery.json", discoveries)

        # Saved-geometry runs do not import or load local model backends.
        stage, scene_name = "grounding", None
        from .grounding.grounding_dino import GroundingDinoDetector, normalize_queries
        from .grounding.postprocess_grounding import process_scene

        detector = GroundingDinoDetector(config.dino_model, device=config.dino_device)
        raw_scenes, clean_scenes, reports = [], [], []
        scene_paths = {scene.name: directory for directory, scene in scenes}
        for discovery in discoveries:
            scene_name = discovery["scene"]
            image_path = scene_paths[scene_name] / "rgb.png"
            queries = normalize_queries(discovery[config.query_source])
            print(f"Grounding {scene_name}: {queries}", flush=True)
            detections = detector.detect(image_path, queries, config.box_threshold, config.text_threshold)
            raw = {"scene": scene_name, "queries": queries, "detections": detections}
            raw_scenes.append(raw)
            save_json(output_dir / "grounding_detections.json", raw_scenes)
            stage = "grounding_cleanup"
            clean, report = process_scene(raw, image_path)
            clean_scenes.append(clean)
            reports.append(report)
            save_json(output_dir / "grounding_detections_clean.json", clean_scenes)
            save_json(output_dir / "grounding_cleanup_report.json", reports)
            if config.debug_visualizations:
                detector.plot_comparison(image_path, detections, clean["detections"],
                                         output_dir / "grounding_debug" / f"{scene_name}.png")
            stage = "grounding"
        del detector
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
        return output_dir / "geometry_results.json"
    except Exception as error:
        save_json(output_dir / "perception.failure.json", {
            "stage": stage, "scene": scene_name,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
