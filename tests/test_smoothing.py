import numpy as np
import pytest

from grainmorph.wadell.smoothing import smooth_boundary


def _dense_smoothing(
    points: np.ndarray,
    perimeter: float,
    alpha_ratio: float,
    beta_ratio: float,
) -> np.ndarray:
    num_points = len(points)
    alpha = alpha_ratio * perimeter
    beta = beta_ratio * perimeter
    coefficients = alpha * np.array([-2, 1, 0, 0]) + beta * np.array([-6, 4, -1, 0])
    distance_matrix = np.fromfunction(
        lambda i, j: np.minimum((i - j) % num_points, (j - i) % num_points),
        (num_points, num_points),
        dtype=int,
    )
    regularization = (
        np.eye(num_points)
        - coefficients[np.minimum(distance_matrix, len(coefficients) - 1)]
    )
    result, *_ = np.linalg.lstsq(regularization, points, rcond=None)
    return result


@pytest.mark.parametrize(
    ("num_points", "perimeter", "alpha_ratio", "beta_ratio"),
    [
        (4, 1.0, 0.0, 1.0),
        (17, 50.0, 0.05, 0.001),
        (64, 200.0, 0.05, 0.001),
    ],
)
def test_fft_smoothing_matches_dense_least_squares(
    num_points, perimeter, alpha_ratio, beta_ratio
):
    points = np.random.default_rng(42).normal(size=(num_points, 2))

    expected = _dense_smoothing(points, perimeter, alpha_ratio, beta_ratio)
    actual = smooth_boundary(points, perimeter, alpha_ratio, beta_ratio)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
