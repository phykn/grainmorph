from numbers import Integral, Real

import numpy as np

from .schema import Mask


def validate_analysis_params(eps: float, target_dim: int) -> tuple[float, int]:
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real):
        raise TypeError("eps must be a real number.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be > 0 and finite.")

    if isinstance(target_dim, (bool, np.bool_)) or not isinstance(target_dim, Integral):
        raise TypeError("target_dim must be an integer.")
    if target_dim <= 0:
        raise ValueError("target_dim must be > 0.")

    return (float(eps), int(target_dim))


def to_binary(mask: np.ndarray) -> Mask:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray.")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D.")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    if np.issubdtype(mask.dtype, np.floating):
        if not np.isfinite(mask).all():
            raise ValueError("mask must not contain NaN or inf.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary with values in {0, 1}.")
        return mask.astype(np.uint8)

    if np.issubdtype(mask.dtype, np.integer):
        if mask.size > 0 and (mask.min() < 0 or mask.max() > 1):
            raise ValueError("mask must be binary with values in {0, 1}.")
        return mask.astype(np.uint8)

    raise ValueError("mask must be binary with values in {0, 1}.")
