import numpy as np


def smooth_boundary(
    points: np.ndarray,
    perimeter: float,
    alpha_ratio: float = 0.05,
    beta_ratio: float = 0.001,
) -> np.ndarray:
    coordinates = points[:-1]
    num_points = len(coordinates)

    if not np.isfinite(perimeter) or perimeter <= 0:
        raise ValueError("perimeter must be > 0 and finite.")
    if not np.isfinite(alpha_ratio) or not np.isfinite(beta_ratio):
        raise ValueError("smoothing ratios must be finite.")

    alpha = alpha_ratio * perimeter
    beta = beta_ratio * perimeter
    kernel = _regularization_kernel(num_points, alpha, beta)

    eigenvalues = np.fft.fft(kernel)
    transformed = np.fft.fft(coordinates, axis=0)
    tolerance = (
        np.finfo(float).eps
        * num_points
        * float(np.max(np.abs(eigenvalues)))
    )
    usable = np.abs(eigenvalues) > tolerance
    solution = np.zeros_like(transformed)
    np.divide(
        transformed,
        eigenvalues[:, None],
        out=solution,
        where=usable[:, None],
    )
    return np.fft.ifft(solution, axis=0).real


def _regularization_kernel(
    num_points: int, alpha: float, beta: float
) -> np.ndarray:
    coefficients = alpha * np.array([-2, 1, 0, 0]) + beta * np.array([-6, 4, -1, 0])
    indices = np.arange(num_points)
    distances = np.minimum(indices, num_points - indices)
    kernel = -coefficients[np.minimum(distances, len(coefficients) - 1)]
    kernel[0] += 1.0
    return kernel
