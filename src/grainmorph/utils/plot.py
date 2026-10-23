from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon


def plot_analysis_results(
    mask: np.ndarray,
    results: Mapping[str, Any],
    figsize: tuple[int, int] = (8, 8),
    title: str = "Analysis Results",
    show: bool = True,
) -> None:

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap="gray", alpha=0.3)

    sphericity_data = results.get("sphericity")
    if sphericity_data:
        _add_circle_overlay(
            ax,
            sphericity_data["inscribed"],
            color="blue",
            label="Inscribed",
            marker="+",
        )
        _add_circle_overlay(
            ax,
            sphericity_data["enclosing"],
            color="red",
            label="Enclosing",
            marker="x",
        )

    aspect_ratio_data = results.get("aspect_ratio")
    if aspect_ratio_data:
        ellipse = aspect_ratio_data.get("ellipse")
        if ellipse:
            _add_ellipse_overlay(ax, ellipse)

    metric_labels = {
        "roundness": "Roundness",
        "circularity": "Circularity",
        "sphericity": "Sphericity",
        "aspect_ratio": "Aspect Ratio",
    }

    stats = []
    for key, label in metric_labels.items():
        data = results.get(key)
        if data and "val" in data:
            stats.append(f"{label}: {data['val']:.4f}")

    ax.set_title(f"{title}\n{', '.join(stats)}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()


def _add_circle_overlay(
    ax, circle_data: Mapping[str, float], *, color: str, label: str, marker: str
) -> None:
    x = circle_data["x"]
    y = circle_data["y"]
    r = circle_data["r"]
    ax.add_patch(Circle((x, y), r, color=color, fill=False, label=label))
    ax.scatter(x, y, color=color, marker=marker)


def _add_ellipse_overlay(ax, ellipse: Mapping[str, Any]) -> None:
    ax.add_patch(
        Ellipse(
            (ellipse["x"], ellipse["y"]),
            ellipse["w"],
            ellipse["h"],
            angle=ellipse["angle"],
            color="yellow",
            fill=False,
            label="Fitted Ellipse",
        )
    )
    if ellipse.get("bbox"):
        ax.add_patch(
            Polygon(
                ellipse["bbox"],
                closed=True,
                color="green",
                fill=False,
                label="Bounding Box",
            )
        )
