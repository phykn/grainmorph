import numpy as np

from .fitting import (
    find_enclosing_circle,
    find_inscribed_circle,
    fit_ellipse,
)
from .mask_ops import crop_mask
from .schema import (
    AspectRatioResult,
    CircularityResult,
    Mask,
    RoundnessResult,
    SphericityResult,
)
from .wadell import compute_roundness as compute_wadell_roundness

# Four-direction Crofton weights indexed by 2x2 pixel pattern, matching
# skimage.measure.perimeter_crofton(image, directions=4).
_CROFTON_WEIGHTS = np.array(
    [
        0,
        np.pi / 4 * (1 + 1 / np.sqrt(2)),
        np.pi / (4 * np.sqrt(2)),
        np.pi / (2 * np.sqrt(2)),
        0,
        np.pi / 4 * (1 + 1 / np.sqrt(2)),
        0,
        np.pi / (4 * np.sqrt(2)),
        np.pi / 4,
        np.pi / 2,
        np.pi / (4 * np.sqrt(2)),
        np.pi / (4 * np.sqrt(2)),
        np.pi / 4,
        np.pi / 2,
        0,
        0,
    ]
)


def compute_roundness(mask: Mask, **params: float) -> RoundnessResult | None:
    value = compute_wadell_roundness(mask, **params)

    if value is None:
        return None

    value = float(np.clip(value, 0.0, 1.0))

    return {"val": value}


def compute_circularity(mask: Mask, *, eps: float = 0.001) -> CircularityResult | None:
    cropped_mask, _, _ = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return None

    perimeter = _crofton_perimeter(cropped_mask)

    if perimeter < eps:
        return None

    area = float(np.count_nonzero(mask))
    value = 4.0 * np.pi * area / perimeter**2
    value = float(np.clip(value, 0.0, 1.0))

    return {"val": value}


def _crofton_perimeter(mask: Mask) -> float:
    padded = np.pad((mask > 0).astype(np.uint8), ((2, 1), (2, 1)))
    patterns = (
        padded[1:, 1:]
        + 2 * padded[:-1, 1:]
        + 4 * padded[1:, :-1]
        + 8 * padded[:-1, :-1]
    )
    counts = np.bincount(patterns.ravel(), minlength=16)
    return float(_CROFTON_WEIGHTS @ counts)


def compute_sphericity(mask: Mask, *, eps: float = 0.001) -> SphericityResult | None:
    inscribed = find_inscribed_circle(mask)
    enclosing = find_enclosing_circle(mask)

    if inscribed is None or enclosing is None:
        return None

    inscribed_radius = inscribed["r"]
    enclosing_radius = enclosing["r"]

    if enclosing_radius < eps:
        return None

    value = float(np.clip(inscribed_radius / enclosing_radius, 0.0, 1.0))

    return {"val": value, "inscribed": inscribed, "enclosing": enclosing}


def compute_aspect_ratio(mask: Mask, *, eps: float = 0.001) -> AspectRatioResult | None:
    ellipse_data = fit_ellipse(mask)

    if ellipse_data is None:
        return None

    if ellipse_data["minor"] < eps:
        return None

    value = ellipse_data["major"] / ellipse_data["minor"]

    return {"val": value, "ellipse": ellipse_data}
