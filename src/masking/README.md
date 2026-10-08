# SAM 2 segmentation from cleaned DINO detections

From the repository root, in a Python 3.11+ environment:

```sh
python -m pip install -r src/masking/requirements.txt
python -m src.masking.sam
```

The default input is **grounding_detections_clean.json**. Every cleaned box
prompts SAM 2 on dataset/<scene>/rgb.png. The default checkpoint is
facebook/sam2.1-hiera-small (SAM 2.1), downloaded on first use. The implementation
uses Transformers' Sam2Model and Sam2Processor, which already fit this project's
Transformers model stack; no separate SAM repository or custom CUDA extension
is required. Torchvision is needed by the SAM 2 image processor. OpenCV, already
a project dependency, supplies the Euclidean distance transform.

Keep the compute container's matching PyTorch/Torchvision builds when they meet
the requirements. For an environment without them, install their appropriate
CPU or CUDA wheels using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/),
then install the requirements above. SAM 2 needs Transformers 4.57.1 or newer;
this requirements file stays on the 4.x series.

```sh
python -m src.masking.sam --scene scene_01 --device cuda:0
python -m src.masking.sam --device cpu
```

CUDA is selected when available, otherwise CPU. Direct execution with
python src/masking/sam.py also works. Run with --help for path, checkpoint,
scene, and validation threshold overrides.

## Mask selection and metadata

All boxes in a scene share one image-encoder pass. SAM 2 returns scored mask
candidates for each prompt. Exactly one is retained per cleaned detection:
the highest predicted quality score among nonempty candidates. Empty candidates
are ineligible; if every candidate is empty or has a nonfinite score, the run
fails with the object ID instead of silently dropping the detection. Masks are
restored to the original image resolution one object's candidates at a time
to limit memory use. Overlapping objects keep independent masks.

All DINO fields are copied unchanged, including id, label, aliases, score, box,
source_ids, needs_review, and review_reasons. The output also has dino_score
(the original score), sam_score, and:

- mask_area: foreground pixel count.
- mask_box: tight mask bounds [x1, y1, x2, y2], with exclusive right/bottom
  coordinates. box remains the original DINO prompt box.
- centroid: arithmetic mean of foreground pixels, in [x, y] order. It can lie
  in a hole or between disconnected parts.
- safe_pixel: an integer foreground pixel furthest from background using
  OpenCV's exact Euclidean distance transform. A zero border is padded around
  the image, so masks touching an image edge also have a finite boundary.
- safe_pixel_clearance: the distance-transform value at that pixel.
- box_mask_fraction, outside_box_fraction, far_outside_box_fraction:
  mask/box validation measurements.
- validation_warnings: the same quality warnings emitted to the console.

The safe pixel is guaranteed inside the mask; it is not the box centre. No
scene/object-specific corrections, relabeling, additional deduplication, or
challenge ground truth are used.

## Validation

Masks must be nonempty, safe pixels must be inside them, and centroids must
be inside the image. Invalid prompt boxes, duplicate scene/object IDs, missing
cleaned metadata, or inconsistent model output counts raise errors.

By default, warn when less than 2% of the DINO box contains mask pixels. Also
warn when more than 10% of mask pixels lie beyond the DINO box expanded by 5%
of its width and height. These are quality warnings, not automatic changes or
removals. Configure them with --min-box-fraction, --max-outside-fraction, and
--outside-box-margin. Masks are never clipped to boxes, so spill remains
measurable and visible. Prompts outside the image are clipped for SAM only;
their original box metadata is preserved and the clipping is reported.

## Outputs

```text
segmentation_results.json
outputs/
  segmentation/
    scene_01/
      grey_tray_1.png
      mug_1.png
      red_cube_1.png
      yellow_gear_1.png
    scene_02/
      ...
  segmentation_debug/
    scene_01.png
    scene_02.png
    ...
```

Masks are compressed binary PNGs (white foreground, black background) at the
original image resolution. The main JSON is a list of scene records with
objects lists, image_path, and any existing scene-level metadata. Object records
contain mask_path and geometry, never full pixel matrices.

Every debug plot shows the original image beside transparent mask overlays,
DINO boxes, unchanged object IDs, and yellow stars at safe pixels.

## Python API and checks

SamSegmenter exposes load_detections, segment_scene, segment_object,
calculate_geometry, validate_mask, save_mask, visualize, and save_results.
Model loading is deferred until segmentation, so geometry and saving can be
used without a checkpoint. segment_scene returns dictionaries with in-memory
NumPy masks under mask; save_results removes those arrays before serialization.

```sh
python -m unittest discover -s tests -p 'test_segmentation.py'
```

The checks use synthetic masks and mocked inference, not challenge ground truth
or downloaded weights. The inference integration check requires PyTorch.
