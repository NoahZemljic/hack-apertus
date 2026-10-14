"""Backend-neutral bounding-box debug plots."""

from pathlib import Path

from PIL import Image


def save_box_plot(
    image_path: Path,
    panels: list[tuple[str, list[dict]]],
    output_path: Path,
    raw_detections: list[dict],
) -> None:
    """Save plots on a headless canvas, keeping source-box colors across panels."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    figure = Figure(figsize=(12 * len(panels), 10))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    palette = colormaps["tab20"]
    color_by_id = {d["id"]: palette(i % 20) for i, d in enumerate(raw_detections)}
    for ax, (title, detections) in zip(axes, panels):
        ax.imshow(image)
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["box"]
            source_id = (detection.get("source_ids") or [detection["id"]])[0]
            color = color_by_id.get(source_id, palette(index % 20))
            review = detection.get("needs_review", False)
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False,
                linewidth=2, edgecolor=color, linestyle="--" if review else "-",
            ))
            ax.text(
                max(0, x1), max(0, y1),
                detection["id"] + (f' {detection["score"]:.2f}' if detection.get("score") is not None else "") + (" ?" if review else ""),
                fontsize=9, color="white",
                bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none"},
            )
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.set_title(f"{title}: {len(detections)} boxes")
        ax.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    figure.clear()

