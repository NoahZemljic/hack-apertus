# DINO boxes and conservative cleanup

Run from the repository root in your DINO environment (PyTorch, Transformers,
Pillow, and Matplotlib):

```sh
python -m src.grounding.grounding_dino
```

Each scene is cleaned immediately after detection. Outputs:

- `grounding_detections.json`: raw DINO boxes and labels.
- `grounding_detections_clean.json`: cleaned boxes with source IDs and review flags.
- `grounding_cleanup_report.json`: removed boxes, reasons, clipping/merge counts,
  and IDs needing review.
- `outputs/grounding/<scene>_boxes.png`: raw boxes.
- `outputs/grounding/<scene>_boxes_clean.png`: cleaned boxes.
- `outputs/grounding/<scene>_comparison.png`: raw and cleaned panels with matching
  colors. Dashed boxes and `?` mark unresolved labels or other review cases.

Clean and plot an existing run without loading DINO or changing the input JSON:

```sh
python -m src.grounding.grounding_dino --input-detections grounding_detections.json
```

Add `--scene scene_06` to process one scene. Direct execution with
`python src/grounding/grounding_dino.py` also works. See `--help` for paths and
threshold overrides.

Cleanup clips boxes to the image, removes invalid boxes, and keeps the strongest
box among nearly identical detections (IoU at least 0.90). It preserves that
box's original label and records every merged label and source ID. Conflicting
duplicate names require review. Huge boxes (over 35% of the image) are removed
only for the small-object classes implicated in the current output: gears,
cubes, mandarins, and components. Other oversized objects are retained for
review; trays, arms, and hands are allowed.

Exact query labels and unknown names keep their original spelling. Repeated
copies of one complete query can be collapsed (`red gear red gear` becomes
`red gear`). Different concatenated queries, such as `mandarin red cube`, keep
their raw label and require review. Neither another instance elsewhere in the
scene nor the length of a query justifies guessing a new object name. Cleanup
cannot identify objects from their pixels or repair a box spanning several
objects.

To give SAM the cleaned boxes:

```sh
python -m src.masking.sam --detections grounding_detections_clean.json
```

Regression checks (Pillow and Matplotlib required; DINO weights are not needed):

```sh
python -m unittest discover -s tests -p 'test_grounding.py'
```
