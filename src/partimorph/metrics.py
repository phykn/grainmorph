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


def compute_roundness(
    mask: Mask,
    *,
    max_dev_thresh: float = 0.3,
    circle_fit_thresh: float = 0.98,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> RoundnessResult | None:
    from .wadell import compute_roundness as compute_roundness_wadell

    value = compute_roundness_wadell(
        mask,
        max_dev_thresh=max_dev_thresh,
        circle_fit_thresh=circle_fit_thresh,
        alpha_ratio=alpha_ratio,
        beta_ratio=beta_ratio,
    )

    if value is None:
        return None

    value = float(np.clip(value, 0.0, 1.0))

    return {"val": value}


def compute_circularity(mask: Mask, *, eps: float = 0.001) -> CircularityResult | None:
    from skimage.measure import perimeter_crofton

    cropped_mask, _, _ = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return None

    perimeter = perimeter_crofton(cropped_mask, 4)

    if perimeter < eps:
        return None

    area = float(np.count_nonzero(mask))
    value = 4.0 * np.pi * area / perimeter**2
    value = float(np.clip(value, 0.0, 1.0))

    return {"val": value}


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
