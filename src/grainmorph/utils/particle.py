import numpy as np

from ..analyzer import analyze_mask
from .shape import compute_polar_vertices, fill_polygon


def create_particle_mask(
    shape,
    center,
    radius,
    sphericity,
    roundness,
    *,
    num_angles=256,
    frequencies=range(2, 9),
    decay=1.0,
    seed=None,
    max_iter=12,
    amp_max=0.45,
    metric_tol=0.001,
    report_tol=0.1,
    return_info=False,
):
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and >= 0.")
    if not (np.isfinite(sphericity) and np.isfinite(roundness)):
        raise ValueError("sphericity and roundness must be finite.")
    if num_angles < 3:
        raise ValueError("num_angles must be >= 3.")

    sphericity = float(np.clip(sphericity, 0.001, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))
    frequencies = np.asarray(frequencies, dtype=int)
    phases = np.random.default_rng(seed).uniform(
        0.0, 2.0 * np.pi, size=len(frequencies)
    )

    amplitude = 0.0
    if radius == 0:
        mask = np.zeros(shape, dtype=bool)
        y, x = round(center[0]), round(center[1])
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = True
    else:
        amplitude = _calibrate_amplitude(
            radius,
            sphericity,
            roundness,
            frequencies,
            phases,
            decay,
            max_iter,
            amp_max,
            metric_tol,
        )
        thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        base, rough = _compute_radii(
            radius, sphericity, thetas, frequencies, phases, decay
        )
        radii = np.maximum(base + amplitude * rough, 0.5)
        mask = fill_polygon(shape, compute_polar_vertices(center, radii, thetas))

    if not return_info:
        return mask

    achieved = _measure_roundness(mask)
    abs_error = abs(roundness - achieved)
    return mask, {
        "sphericity_target": sphericity,
        "roundness_target": roundness,
        "roundness_achieved": achieved,
        "abs_error": abs_error,
        "report_tol": float(report_tol),
        "target_met": abs_error <= report_tol,
        "amplitude": amplitude,
        "frequencies": frequencies.tolist(),
        "phases": phases.tolist(),
    }


def _calibrate_amplitude(
    radius,
    sphericity,
    roundness,
    frequencies,
    phases,
    decay,
    max_iter,
    amp_max,
    metric_tol,
):
    thetas = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    base, rough = _compute_radii(radius, sphericity, thetas, frequencies, phases, decay)
    scale = min(1.0, 128.0 / radius)
    size = int(np.ceil((radius * scale * (1.0 + amp_max) + 2) * 2))

    def measure(amplitude):
        radii = np.maximum((base + amplitude * rough) * scale, 0.5)
        vertices = compute_polar_vertices((size / 2, size / 2), radii, thetas)
        return _measure_roundness(fill_polygon((size, size), vertices))

    # Roughness only lowers roundness, so a smooth shape below the target stays smooth.
    amplitude = 0.0
    achieved = measure(amplitude)
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


def _compute_radii(radius, sphericity, thetas, frequencies, phases, decay):
    minor = radius * sphericity
    base = (
        radius
        * minor
        / np.sqrt((minor * np.cos(thetas)) ** 2 + (radius * np.sin(thetas)) ** 2)
    )

    weights = 1.0 / np.maximum(frequencies, 1.0) ** decay
    rough = np.sum(
        weights[:, None] * np.cos(frequencies[:, None] * thetas + phases[:, None]),
        axis=0,
    )
    peak = np.abs(rough).max()
    if peak > 1e-12:
        rough = rough / peak
    return base, rough * radius


def _measure_roundness(mask):
    results = analyze_mask(
        mask, use_aspect_ratio=False, use_circularity=False, use_sphericity=False
    )
    return results["roundness"]["val"] if results and results["roundness"] else 0.0
