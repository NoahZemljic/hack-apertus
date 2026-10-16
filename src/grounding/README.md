# Qwen boxes and conservative cleanup

The supported pipeline uses Qwen3-VL-32B to identify visible objects and their
bounding boxes, then cleans those boxes before SAM segmentation. Run it through
[the project launcher](../../README.md#getting-started-on-clariden).

Artifacts are saved beside the submission under `$HOME/hack-apertus/outputs`
by default:

- `grounding_detections.json`: raw Qwen boxes and labels.
- `grounding_detections_clean.json`: cleaned boxes with source IDs and review flags.
- `grounding_cleanup_report.json`: dropped boxes, clipping and merge counts,
  and IDs needing review.
- `grounding_debug/<scene>.png`: comparison plots when `--debug-visualizations` is set.

To clean saved detections without loading models:

```bash
python -m src.grounding.postprocess_grounding \
  --input "$HOME/hack-apertus/outputs/final/grounding_detections.json" \
  --output "$HOME/hack-apertus/outputs/final/grounding_detections_clean.json" \
  --report "$HOME/hack-apertus/outputs/final/grounding_cleanup_report.json" \
  --dataset dataset
```

Cleanup clips boxes to the image, removes invalid boxes, and merges nearly
identical detections (IoU at least 0.90). Qwen confidence remains `null`; input
order determines which duplicate is retained. Conflicting duplicate labels
require review. Boxes covering over 35% of the image are removed only for gears,
cubes, mandarins and components. Other oversized objects remain for review;
trays, arms and hands are allowed. Cleanup preserves labels and cannot repair
object identity from pixels.

Check the project environment without GPU weights:

```bash
python -m src.check_environment --profile base
```
