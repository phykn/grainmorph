import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse

import grainmorph as gm


@pytest.mark.parametrize("angle", [25, -35])
def test_overlays_align_with_rotated_particle(angle):
    mask = np.zeros((160, 180), dtype=np.uint8)
    cv2.ellipse(mask, (90, 80), (50, 18), angle, 0, 360, 1, -1)
    results = gm.analyze_mask(mask, use_roundness=False)

    gm.utils.plot_analysis_results(mask, results, title="Particle", show=False)
    fig = plt.gcf()
    try:
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["Inscribed", "Enclosing", "Fitted Ellipse", "Bounding Box"]
        assert ax.get_title().startswith("Particle\nCircularity: 0.")
        assert "Roundness" not in ax.get_title()

        ellipse = next(patch for patch in ax.patches if type(patch) is Ellipse)
        points = ellipse.get_patch_transform().transform(ellipse.get_path().vertices)
        y, x = np.nonzero(mask)
        _, particle_axes = np.linalg.eigh(np.cov(np.column_stack([x, y]).T))
        _, overlay_axes = np.linalg.eigh(np.cov(points.T))
        assert abs(particle_axes[:, -1] @ overlay_axes[:, -1]) > 0.995
    finally:
        plt.close(fig)
