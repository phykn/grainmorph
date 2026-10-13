import cv2
import numpy as np

from ..fitting import find_inscribed_circle
from ..mask_ops import crop_mask
from ..schema import Mask
from .boundary import extract_boundary
from .corner import compute_corner_radii
from .discretize import discretize_boundary
from .smoothing import smooth_boundary


def compute_roundness(
    mask: Mask,
    max_dev_thresh: float = 0.3,
    circle_fit_thresh: float = 0.98,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> float | None:
    mask_cropped, _, _ = crop_mask(mask, pad=1)
    if mask_cropped.size == 0:
        return None

    inscribed = find_inscribed_circle(mask_cropped)
    if inscribed is None or inscribed["r"] < 1e-6:
        return None

    max_radius = inscribed["r"]
    max_radius_pos = np.array([inscribed["x"], inscribed["y"]])

    boundary = extract_boundary(mask_cropped)
    if len(boundary) < 4:
        return None

    perimeter = float(
        cv2.arcLength(boundary[:-1].astype(np.float32).reshape(-1, 1, 2), closed=True)
    )
    if perimeter < 1e-6:
        return None

    smoothed_boundary = smooth_boundary(
        boundary,
        perimeter=perimeter,
        alpha_ratio=alpha_ratio,
        beta_ratio=beta_ratio,
    )

    keypoints = discretize_boundary(smoothed_boundary, max_dev_thresh)
    if len(keypoints) < 3:
        return None

    radii = compute_corner_radii(
        keypoints,
        max_radius,
        max_radius_pos,
        circle_fit_thresh,
    )
    if len(radii) == 0:
        return None

    return float(np.mean(radii) / max_radius)
