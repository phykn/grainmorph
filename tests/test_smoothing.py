import cv2
import numpy as np
import pytest

from grainmorph.wadell.smoothing import smooth_boundary


@pytest.mark.parametrize(
    ("num", "alpha_ratio", "beta_ratio"),
    [(4, 0.0, 1.0), (17, 0.05, 0.001), (64, 0.05, 0.001)],
)
def test_smoothing_matches_dense_least_squares(num, alpha_ratio, beta_ratio):
    points = np.random.default_rng(42).normal(size=(num, 2))
    perimeter = cv2.arcLength(points.astype(np.float32).reshape(-1, 1, 2), closed=True)
    coefs = alpha_ratio * perimeter * np.array([-2, 1, 0, 0])
    coefs = coefs + beta_ratio * perimeter * np.array([-6, 4, -1, 0])
    i, j = np.indices((num, num))
    distance = np.minimum((i - j) % num, (j - i) % num)
    expected, *_ = np.linalg.lstsq(
        np.eye(num) - coefs[np.minimum(distance, 3)], points, rcond=None
    )

    actual = smooth_boundary(points, alpha_ratio, beta_ratio)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_smoothing_rejects_non_finite_ratios():
    with pytest.raises(ValueError, match="finite"):
        smooth_boundary(np.zeros((8, 2)), np.nan, 0.001)
