def plot_analysis_results(
    mask, results, figsize=(8, 8), title="Analysis Results", show=True
):
    # Imported here so that importing grainmorph does not load matplotlib.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Polygon

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap="gray", alpha=0.3)

    sphericity = results.get("sphericity")
    if sphericity:
        for key, color, label, marker in (
            ("inscribed", "blue", "Inscribed", "+"),
            ("enclosing", "red", "Enclosing", "x"),
        ):
            circle = sphericity[key]
            center = (circle["x"], circle["y"])
            ax.add_patch(
                Circle(center, circle["r"], color=color, fill=False, label=label)
            )
            ax.scatter(*center, color=color, marker=marker)

    aspect_ratio = results.get("aspect_ratio")
    if aspect_ratio:
        ellipse = aspect_ratio["ellipse"]
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
        ax.add_patch(
            Polygon(
                ellipse["bbox"],
                closed=True,
                color="green",
                fill=False,
                label="Bounding Box",
            )
        )

    stats = [
        f"{label}: {results[key]['val']:.4f}"
        for key, label in (
            ("roundness", "Roundness"),
            ("circularity", "Circularity"),
            ("sphericity", "Sphericity"),
            ("aspect_ratio", "Aspect Ratio"),
        )
        if results.get(key)
    ]
    ax.set_title(f"{title}\n{', '.join(stats)}")
    if ax.patches:
        ax.legend(loc="upper right")

    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
