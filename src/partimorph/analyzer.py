import numpy as np

from .mask_ops import prepare_mask
from .metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)
from .schema import AnalysisResult
from .validation import to_binary, validate_analysis_params


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

    if not (use_aspect_ratio or use_roundness or use_circularity or use_sphericity):
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

    prepared.restore_results(results)

    return results
