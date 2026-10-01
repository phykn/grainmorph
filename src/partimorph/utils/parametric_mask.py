import numpy as np

from ..metrics import compute_roundness
from .geometry import create_poly_mask, polar_vertices


def create_particle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    sphericity: float,
    roundness: float,
    *,
    num_angles: int = 256,
    frequencies: list[int] | np.ndarray | None = None,
    decay: float = 1.0,
    seed: int | None = None,
    max_iter: int = 20,
    amp_max: float = 0.45,
    metric_tol: float = 0.001,
    report_tol: float = 0.1,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generates a synthetic particle mask with target morphological features.

    Uses a Fourier-based approach to generate a closed polygon-based mask that
    matches specified sphericity and roundness values through iterative optimization
    of the roughness amplitude.

    Args:
        shape: Dimensions of the output mask (height, width).
        center: Center coordinates of the particle (y, x).
        radius: Base radius of the particle.
        sphericity: Target sphericity (0 to 1).
        roundness: Target roundness (0 to 1).
        num_angles: Number of vertices in the generated polygon.
        frequencies: List of frequencies used to generate surface roughness.
        decay: Frequency decay rate of the roughness signal.
        seed: Random seed for phase generation.
        max_iter: Max iterations for roundness calibration.
        amp_max: Maximum allowed amplitude for the roughness signal.
        metric_tol: Internal precision for reaching the target roundness.
        report_tol: Error threshold for the 'target_met' report flag.
        return_info: If True, returns a metadata dictionary along with the mask.

    Returns:
        The binary mask as a numpy array, or a tuple of (mask, info_dict).
    """
    try:
        radius = float(radius)
        sphericity = float(sphericity)
        roundness = float(roundness)
    except (TypeError, ValueError):
        raise ValueError(
            "radius, sphericity, and roundness must be finite numbers."
        ) from None

    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and >= 0.")
    if not np.isfinite(sphericity) or not np.isfinite(roundness):
        raise ValueError("sphericity and roundness must be finite.")

    sphericity = float(np.clip(sphericity, 0.001, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))
    if num_angles < 3:
        raise ValueError("Invalid angles")

    frequencies_arr = np.asarray(
        frequencies if frequencies is not None else np.arange(2, 9), dtype=int
    )
    phases = np.random.default_rng(seed).uniform(
        0.0, 2.0 * np.pi, size=len(frequencies_arr)
    )
    cy, cx = center
    amplitude = 0.0

    if radius == 0.0:
        mask = np.zeros(shape, dtype=bool)
        iy, ix = round(cy), round(cx)
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            mask[iy, ix] = True
    else:
        amplitude = _calibrate_amplitude(
            radius=radius,
            sphericity=sphericity,
            roundness=roundness,
            frequencies=frequencies_arr,
            phases=phases,
            decay=decay,
            max_iter=max_iter,
            amp_max=amp_max,
            metric_tol=metric_tol,
        )
        thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        base_radii, roughness_radii = _radial_components(
            radius, sphericity, thetas, frequencies_arr, phases, decay
        )
        radii = np.maximum(base_radii + amplitude * roughness_radii, 0.5)
        vertices = polar_vertices((cy, cx), radii, thetas)
        mask = create_poly_mask(shape, vertices).astype(bool)

    if not return_info:
        return mask

    measured = compute_roundness(mask.astype(np.uint8, copy=False)) if radius else None
    achieved = float(measured["val"]) if measured else 0.0
    abs_error = float(abs(roundness - achieved))
    info = {
        "sphericity_target": sphericity,
        "roundness_target": roundness,
        "roundness_achieved": achieved,
        "abs_error": abs_error,
        "report_tol": float(report_tol),
        "target_met": bool(abs_error <= report_tol),
        "amplitude": float(amplitude),
        "frequencies": frequencies_arr.tolist(),
        "phases": phases.tolist(),
    }
    return mask, info


def _calibrate_amplitude(
    *,
    radius: float,
    sphericity: float,
    roundness: float,
    frequencies: np.ndarray,
    phases: np.ndarray,
    decay: float,
    max_iter: int,
    amp_max: float,
    metric_tol: float,
) -> float:
    thetas = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    base_radii, roughness_radii = _radial_components(
        radius, sphericity, thetas, frequencies, phases, decay
    )
    scale = min(1.0, 128.0 / radius)
    size = int(np.ceil((radius * scale * (1.0 + amp_max) + 2) * 2))
    shape = (size, size)
    center = size / 2.0

    def measure(amplitude: float) -> float:
        radii = (base_radii + amplitude * roughness_radii) * scale
        vertices = polar_vertices((center, center), np.maximum(radii, 0.5), thetas)
        result = compute_roundness(create_poly_mask(shape, vertices))
        return float(result["val"]) if result else 0.0

    achieved = measure(0.0)
    amplitude = 0.0
    if achieved > roundness + metric_tol:
        low, high = 0.0, float(amp_max)
        for _ in range(max_iter):
            mid = (low + high) / 2
            measured = measure(mid)
            if abs(measured - roundness) < abs(achieved - roundness):
                achieved, amplitude = measured, mid
            if abs(measured - roundness) <= metric_tol:
                break
            if measured > roundness:
                low = mid
            else:
                high = mid

    return amplitude


def _radial_components(
    radius: float,
    sphericity: float,
    thetas: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_radii = ellipse_radius(
        radius, radius * sphericity, np.cos(thetas), np.sin(thetas)
    )
    roughness_radii = roughness_signal(thetas, frequencies, phases, decay) * radius
    return base_radii, roughness_radii


def ellipse_radius(
    a: float, b: float, cos_t: np.ndarray, sin_t: np.ndarray
) -> np.ndarray:
    denom = np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
    return a * b / np.maximum(denom, 1e-12)


def roughness_signal(
    thetas: np.ndarray, frequencies: np.ndarray, phases: np.ndarray, decay: float
) -> np.ndarray:
    weights = 1.0 / np.maximum(frequencies.astype(float), 1.0) ** decay
    components = weights[:, None] * np.cos(
        frequencies[:, None] * thetas[None, :] + phases[:, None]
    )
    signal = np.sum(components, axis=0)
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs > 1e-12 else signal
