import numpy as np

from .mask_ops import PreparedMask, prepare_mask
from .metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)
from .schema import AnalysisResult
from .validation import to_binary, validate_analysis_params


def _restore_results(
    results: AnalysisResult,
    prepared: PreparedMask,
) -> None:
    scale = prepared.scale

    if results.get("sphericity"):
        for key in ("inscribed", "enclosing"):
            circle_data = results["sphericity"][key]
            circle_data["x"] = float(
                circle_data["x"] * scale + prepared.offset_x
            )
            circle_data["y"] = float(
                circle_data["y"] * scale + prepared.offset_y
            )
            circle_data["r"] = float(circle_data["r"] * scale)

    if results.get("aspect_ratio"):
        ellipse = results["aspect_ratio"]["ellipse"]
        ellipse["x"] = float(ellipse["x"] * scale + prepared.offset_x)
        ellipse["y"] = float(ellipse["y"] * scale + prepared.offset_y)
        ellipse["major"] = float(ellipse["major"] * scale)
        ellipse["minor"] = float(ellipse["minor"] * scale)
        ellipse["w"] = float(ellipse["w"] * scale)
        ellipse["h"] = float(ellipse["h"] * scale)
        ellipse["bbox"] = [
            [
                float(pt[0] * scale + prepared.offset_x),
                float(pt[1] * scale + prepared.offset_y),
            ]
            for pt in ellipse["bbox"]
        ]


def analyze_mask(
    mask: np.ndarray,
    *,
    use_aspect_ratio: bool = True,
    use_roundness: bool = True,
    use_circularity: bool = True,
    use_sphericity: bool = True,
    roundness_params: dict[str, float] | None = None,
    eps: float = 0.001,
    target_dim: int = 384,
) -> AnalysisResult | None:
    eps, target_dim = validate_analysis_params(eps, target_dim)

    mask_binary = to_binary(mask)
    if not np.any(mask_binary):
        return None

    if not (
        use_aspect_ratio or use_roundness or use_circularity or use_sphericity
    ):
        return {}

    prepared = prepare_mask(mask_binary, target_dim=target_dim)
    final_mask = prepared.mask

    results: AnalysisResult = {}
    if use_aspect_ratio:
        results["aspect_ratio"] = compute_aspect_ratio(final_mask, eps=eps)
    if use_roundness:
        results["roundness"] = compute_roundness(final_mask, **(roundness_params or {}))
    if use_circularity:
        results["circularity"] = compute_circularity(final_mask, eps=eps)
    if use_sphericity:
        results["sphericity"] = compute_sphericity(final_mask, eps=eps)

    _restore_results(results, prepared)

    return results
