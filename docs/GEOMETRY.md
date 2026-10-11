# Deterministic geometry after SAM

Run from the repository root in Python 3.11+:

```sh
python -m src.geometry
```

This reads `outputs/grounded_pipeline/segmentation_results.json` and its **SAM mask PNGs**, with images
from each record's `image_path` (normally `dataset/<scene>/rgb.png`). It performs
no model loading, LLM calls, downloads, or ground-truth access. NumPy and OpenCV
are already dependencies in `pyproject.toml`; no new dependency is required.
For a fresh environment, `python -m pip install 'numpy>=2,<3' 'opencv-python-headless>=4.10,<6'`
installs these two dependencies. This module also runs with networking disabled.

```sh
python -m src.geometry --scene scene_06
python -m src.geometry --input /path/to/segmentation_results.json --base-dir /path/to/project
python -m src.geometry --config geometry_thresholds.json
python -m src.geometry --no-visualizations
python -m unittest discover -s tests -p 'test_geometry.py'
```

Relative mask/image paths resolve against the input JSON's directory, or
`--base-dir` when supplied. `--dataset` overrides recorded image paths with
`<dataset>/<scene>/rgb.png`. Output paths resolve against the working directory.
`--scene` can be repeated; unknown names raise an error. A subset run writes
only the selected scenes to its JSON outputs, so use separate `--output` and
`--debug-output` paths when keeping a full run. Old visualization files are not
deleted. Use `--help` for all path options.

## Relation definitions

Coordinates are full-resolution `[x, y]` pixels, origin top left, with y
increasing downward. Every relation except the support hypotheses describes
the **image plane**, not a statement about depth or physical contact.

| Relation | Evidence and default rule |
| --- | --- |
| `left_of` / `right_of` | Centroid x separation exceeds both 8 reference pixels and 20% of mean tight-mask width. The actual foreground y projections must overlap by at least 25% of the smaller projection. Separation is at most 3 mean widths. |
| `above` / `below` | Equivalent centroid y test using heights and overlapping foreground x projections. |
| `overlaps` | Actual mask intersection is at least 16 reference pixels squared and covers at least 2% of the smaller mask. Debug output includes intersection, union, IoU, and both directed coverage ratios. |
| `inside` / `contains` | At least 95% of A's actual mask lies in B's actual mask. Alternatively, at least 98% lies within B's closed outer-contour region, including its holes. B's enclosed region must be at least 1.25 times A's enclosed region, preventing mutual containment of near-identical masks. Each edge identifies `actual_mask` or `enclosed_mask_region` evidence. |
| `touching` | At least 3% of either external contour lies within 3 reference pixels of the other contour. Actual overlap is at most 10% of the smaller mask. Contained pairs are excluded. This is boundary proximity in the image, not proven physical contact. |
| `on` / `under` | Reviewable hypotheses from one of the two patterns below. Neither is automatically promoted to physical support. |

Directional output keeps the union of each object's single nearest eligible
neighbor per direction, plus inverse edges. A nearest neighbor chooses by
centroid separation, with IDs breaking ties. This permits more than one inverse
neighbor when several objects select the same neighbor. Nested or near-fully
overlapping pairs are excluded from directional inference. Unaligned or very
distant pairs produce no directional edges. Raw measurements still cover every
unordered pair for tuning; this does not create a relation for every pair.

Hole filling uses each external mask contour, **not a convex hull**. A closed
tray silhouette can enclose an object removed from its mask by occlusion. An
open U-shaped mask does not contain everything in its convex hull. Actual,
hole-filled, and convex-hull coverage are all recorded separately; convex hulls
are diagnostic and supply a shape-solidity measure. Image enclosure must not
be mistaken for an object being physically inside a container, nor should
containment create a `supports` edge.

All pixel thresholds scale by `max(image_width, image_height) / 2048`.
Touch tolerance has a one-pixel minimum. For the current 2448×2048 images, the
minimum directional separation is 9.56 pixels (or the larger size-based
threshold), touch tolerance is 3.59 pixels, and overlap needs at least 23 pixels.

### Conservative support hypotheses

The module accepts a proposed A-on-B pattern only if all its checks pass:

- **Opposing edges:** A's centroid is above B's beyond the vertical dead zone;
  x projections overlap at least 65%; centroid x offset is at most 25% of mean
  width; area ratio A/B is 0.25–4; at least 15% of the smaller width has A's
  actual bottom mask pixels meeting B's actual top mask pixels within touch
  tolerance. These are per-column pixel profiles, not bounding-box edges.
  The pair cannot be contained, and actual overlap is at most 35% of the smaller
  mask. Adjacent visible masks can qualify without overlapping foreground.
- **Offset overlap:** A/B area ratio is 0.10–0.65; at least 80% of A overlaps B's
  actual mask; at least 15% of A's external boundary is near B's external
  boundary; absolute centroid y offset is 12–50% of B's height, and x offset is
  at most 30% of B's width. Both tight-mask aspect ratios must be at most 1.6,
  and B's mask/convex-hull area (solidity) must be at least 0.70.

The second pattern considers either sign of the image y offset: an object
near the image bottom may be physically higher in a top-down photo. The numeric
centroid ordering remains in the debug output. The smaller overlapping mask
is the proposed foreground object; single-view masks cannot establish that
depth ordering with certainty. Centered containment alone is insufficient.

Each object gets at most one proposed support. If the two strongest pattern
scores differ by at most 10% of the strongest score, the module abstains.
Opposing-edge score is contact-column fraction; offset-overlap score is source
coverage × source boundary-contact fraction × target solidity. Scores are
ranking measures, not calibrated confidence probabilities. Support cycles are
also rejected. Debug output retains all individual checks and selection reasons.

These thresholds are conservative starting values, not calibrated against
challenge answers. They can miss real stacks with weak or occluded boundaries.
A single top-down RGB view cannot reliably determine height, occlusion order,
or whether an object rests on another object. Every `on`/`under` edge therefore
has `scope: "physical_hypothesis"` and `needs_review: true`. Existing ambiguous
detection flags are retained and propagated to affected edges; labels are
never changed or interpreted to invent geometry.

## Configuration and Python API

All inference thresholds live in the frozen `GeometryConfig` dataclass near
the top of `src/geometry.py`. A config file may override any subset:

```json
{
  "inside_mask_fraction": 0.97,
  "touch_tolerance_pixels": 2.0,
  "direction_neighbors": 1
}
```

The complete effective config is included in `geometry_debug.json`. Unknown
fields and invalid values raise errors. `GeometryReasoner` exposes
`load_detections`, `load_mask`, `measure_boundaries`, `measure_pair`,
`measure_support_profile`, `infer_relations`, `analyze_scene`, `visualize`, and
`save_results`. `analyze_scene` returns `(result, debug)` without mutating input.

Masks must be binary, nonempty, and match the image size. Saved area, centroid,
optional tight mask bounds, and safe-pixel membership are checked against the
actual PNG. Stale metadata raises an error instead of being silently rewritten.
Centroids may validly fall in a hole; safe pixels must be foreground pixels.
The original DINO box is retained, and mask geometry uses independently
calculated tight mask bounds. No box clipping, relabeling, mask edits, or
object-specific exceptions occur here.

## Outputs and the future scene-graph builder

```text
geometry_results.json
geometry_debug.json
outputs/
  geometry_debug/
    scene_01.png
    scene_01_directions.png
    scene_02.png
    scene_02_directions.png
    ...
```

`geometry_results.json` is a list of scene records. All existing scene/object
metadata survives, including IDs, labels, aliases, boxes, centroids, safe
pixels, mask areas, scores, mask paths, and review flags. New fields are:

- Scene `image`: width and height; `geometry_version`: output contract version.
- Scene `relations`: image-plane edges with `source`, `target`, `type`,
  `evidence`, `scope`, `needs_review`, `input_needs_review`, and
  `measurements_index`.
- Scene `relation_candidates`: separate `on` / inverse `under` hypotheses,
  with explicit review reasons.
- Object `geometric_relations`: lists of IDs per pixel relation, including
  symmetric overlap/touching links and directional/containment inverses.
- Object `geometric_candidates`: separate `on` and `under` ID lists.

Symmetric overlap/touching pairs have one scene-level edge and two object-level
links. Every edge's `measurements_index` identifies the corresponding record
in `geometry_debug.json` → `scenes[matching_scene].pairs`. Debug records also
contain object shape statistics, effective pixel scale/tolerance, all pairwise
coverage/boundary/profile measurements, and accepted/rejected stacking checks.
Neither JSON contains mask matrices.

Structural visualizations show transparent masks, IDs, safe-pixel stars,
centroids, and numbered containment/overlap/contact arrows. Magenta `on?`
arrows show support hypotheses. Separate directional images show nearest
left/above arrows, with inverses omitted for readability. Legends identify
each arrow's source, relation, and target.

This is an **intermediate geometry contract**, not the strict planner schema
in `schema/scene_graph.schema.json`. A subsequent scene-graph builder can use
`id` unchanged, convert `box` to the planner's integer `bbox_2d` and use
`safe_pixel` as `pick_point_2d`, and derive semantic object properties separately.
It must validate image bounds and point membership when constructing that box;
a SAM mask can extend outside its original DINO prompt. It must resolve projected
containment against object kinds and select a support acceptance policy before
writing `inside` / `on_top_of`; `src.scene_graph.derive_support_relations` then
derives the inverse support lists. The existing `run.py` pipeline remains
separate. The new `run_grounded.py` entry point runs all perception stages and
planning; `--geometry-input geometry_results.json` reuses saved artifacts.
See [the grounded planning pipeline](GEOMETRY_PIPELINE.md).
That builder accepts selected support hypotheses as first-pass estimates by
default, with an option to omit them. This geometry module itself adds no LLM
decisions or automatic support claims.
