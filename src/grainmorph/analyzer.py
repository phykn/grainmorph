from numbers import Integral

import numpy as np

from .mask import prepare_mask, restore_coordinates
from .metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)


def analyze_mask(
    mask,
    *,
    use_aspect_ratio=True,
    use_roundness=True,
    use_circularity=True,
    use_sphericity=True,
    roundness_params=None,
    eps=0.001,
    target_dim=384,
):
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError("eps must be > 0 and finite.")
    if isinstance(target_dim, bool) or not isinstance(target_dim, Integral):
        raise TypeError("target_dim must be an integer.")
    if target_dim <= 0:
        raise ValueError("target_dim must be > 0.")
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray.")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D.")
    if mask.dtype.kind not in "biuf" or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask must be binary with values in {0, 1}.")

    mask = mask.astype(np.uint8)
    if not mask.any():
        return None

    mask, scale, offset_x, offset_y = prepare_mask(mask, target_dim)

    results = {}
    if use_aspect_ratio:
        results["aspect_ratio"] = compute_aspect_ratio(mask, eps)
    if use_roundness:
        results["roundness"] = compute_roundness(mask, **(roundness_params or {}))
    if use_circularity:
        results["circularity"] = compute_circularity(mask, eps)
    if use_sphericity:
        results["sphericity"] = compute_sphericity(mask, eps)

    restore_coordinates(results, scale, offset_x, offset_y)
    return results
