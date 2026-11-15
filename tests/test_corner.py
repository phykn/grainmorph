import cv2
import numpy as np
import pytest

import grainmorph as gm
from grainmorph.mask import prepare_mask
from grainmorph.metrics import find_inscribed_circle
from grainmorph.wadell.corner import (
    _fit_circle,
    _fit_circles,
    _select_convex_points,
    compute_corner_radii,
)
from grainmorph.wadell.discretize import discretize_boundary
from grainmorph.wadell.smoothing import smooth_boundary


def _compute_corner_radii_by_loop(keypoints, r_max, r_max_pos, circle_fit_thresh):
    convex = _select_convex_points(keypoints).astype(np.float32)
    keypoints = keypoints.astype(np.float32)

    radii = []
    start = 0
    while start < len(convex) - 1:
        for end in range(len(convex), start + 2, -1):
            subset = convex[start:end]
            radius, center = _fit_circle(subset)
            if not radius > 1e-06:
                continue

            dist_center = np.linalg.norm(center - r_max_pos)
            dist_mean = np.linalg.norm(subset - r_max_pos, axis=1).mean()
            if dist_center < dist_mean and radius < r_max:
                dist_all = np.linalg.norm(keypoints - center, axis=1)
                if np.all(dist_all / radius > circle_fit_thresh):
                    radii.append(radius)
                    start = end - 1
                    break
        start += 1

    return np.array(radii)


def _create_masks():
    shape = (400, 400)
    yield gm.utils.create_circle_mask(shape, (200, 200), 180)
    yield gm.utils.create_ellipse_mask(shape, (200, 200), 180, 90)
    yield gm.utils.create_square_mask(shape, (50, 50), 300)
    yield gm.utils.create_triangle_mask(shape, (40, 60), (360, 120), (180, 370))
    yield gm.utils.create_star_mask(shape, (200, 200), 180, 80)
    for seed, sphericity, roundness in [(0, 0.9, 0.6), (1, 0.8, 0.3), (2, 0.7, 0.2)]:
        yield gm.utils.create_particle_mask(
            shape, (200, 200), 170, sphericity, roundness, seed=seed, max_iter=3
        )


@pytest.mark.parametrize(
    ("max_dev_thresh", "circle_fit_thresh"), [(0.3, 0.98), (1.0, 0.9)]
)
def test_corner_radii_match_loop_implementation(max_dev_thresh, circle_fit_thresh):
    for i, mask in enumerate(_create_masks()):
        mask, *_ = prepare_mask(mask.astype(np.uint8), 384)
        inscribed = find_inscribed_circle(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = contours[0].reshape(-1, 2).astype(np.float64)
        keypoints = discretize_boundary(
            smooth_boundary(boundary, 0.05, 0.001), max_dev_thresh
        )
        args = (
            keypoints,
            inscribed["r"],
            np.array([inscribed["x"], inscribed["y"]]),
            circle_fit_thresh,
        )

        actual = compute_corner_radii(*args)

        expected = _compute_corner_radii_by_loop(*args)
        assert len(actual) == len(expected) > 0, i
        np.testing.assert_allclose(actual, expected, rtol=1e-9, err_msg=str(i))


@pytest.mark.parametrize("noise", [0.0, 1e-3, 1e-1])
def test_prefix_circle_fits_match_least_squares(noise):
    rng = np.random.default_rng(0)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, 40))
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 50.0 + [120.0, 80.0]
    t = np.linspace(0.0, 30.0, 12)
    line = np.stack([t, 0.5 * t], axis=1)

    for points in (circle, line):
        points = (points + rng.normal(0.0, noise, points.shape)).astype(np.float32)
        sizes = np.arange(3, len(points) + 1)

        radius, center = _fit_circles(points, sizes)

        for i, size in enumerate(sizes):
            expected_radius, expected_center = _fit_circle(points[:size])
            np.testing.assert_allclose(radius[i], expected_radius, rtol=1e-9)
            np.testing.assert_allclose(center[i], expected_center, rtol=1e-9)
