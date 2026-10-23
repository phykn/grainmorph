import cv2
import numpy as np
import pytest

from partimorph.fitting import find_inscribed_circle
from partimorph.mask_ops import crop_mask, prepare_mask
from partimorph.utils.create_mask import (
    create_circle_mask,
    create_ellipse_mask,
    create_pentagon_mask,
    create_square_mask,
    create_star_mask,
    create_triangle_mask,
)
from partimorph.utils.parametric_mask import create_particle_mask
from partimorph.wadell.boundary import extract_boundary
from partimorph.wadell.corner import (
    _fit_circle,
    _fit_circles,
    _select_convex_points,
    compute_corner_radii,
)
from partimorph.wadell.discretize import discretize_boundary
from partimorph.wadell.smoothing import smooth_boundary


def _loop_corner_radii(keypoints, r_max, r_max_pos, circle_fit_thresh):
    convex_points = _select_convex_points(keypoints).astype(np.float32)
    keypoints = keypoints.astype(np.float32)
    num_points = len(convex_points)

    radii = []
    start = 0

    while start < num_points - 1:
        for end in range(num_points, start + 2, -1):
            point_subset = convex_points[start:end]
            radius, center = _fit_circle(point_subset)

            if radius <= 1e-06 or not np.isfinite(radius):
                continue

            dist_center = np.linalg.norm(center - r_max_pos)
            dist_boundary = np.linalg.norm(point_subset - r_max_pos, axis=1).mean()

            if dist_center < dist_boundary and radius < r_max:
                dist_all = np.linalg.norm(keypoints - center, axis=1)

                if np.all(dist_all / radius > circle_fit_thresh):
                    radii.append(radius)
                    start = end - 1
                    break

        start += 1

    return np.array(radii)


def _corner_inputs(mask, max_dev_thresh=0.3):
    mask = prepare_mask(mask.astype(np.uint8), target_dim=384).mask
    mask, _, _ = crop_mask(mask, pad=1)
    inscribed = find_inscribed_circle(mask)
    boundary = extract_boundary(mask)
    perimeter = float(
        cv2.arcLength(boundary[:-1].astype(np.float32).reshape(-1, 1, 2), closed=True)
    )
    smoothed = smooth_boundary(boundary, perimeter=perimeter)
    keypoints = discretize_boundary(smoothed, max_dev_thresh)
    return keypoints, inscribed["r"], np.array([inscribed["x"], inscribed["y"]])


def _masks():
    shape = (400, 400)
    yield "circle", create_circle_mask(shape, (200, 200), 180)
    yield "ellipse", create_ellipse_mask(shape, (200, 200), 180, 90)
    yield "square", create_square_mask(shape, (50, 50), 300)
    yield "triangle", create_triangle_mask(shape, (40, 60), (360, 120), (180, 370))
    yield "pentagon", create_pentagon_mask(shape, (200, 200), 170)
    yield "star", create_star_mask(shape, (200, 200), 180, 80)
    for seed in range(6):
        for sphericity, roundness in [(0.9, 0.6), (0.8, 0.3), (0.7, 0.2)]:
            mask = create_particle_mask(
                shape, (200, 200), 170, sphericity, roundness, seed=seed, max_iter=3
            )
            yield f"particle-{seed}-{sphericity}-{roundness}", mask


@pytest.mark.parametrize("max_dev_thresh", [0.3, 1.0])
@pytest.mark.parametrize("circle_fit_thresh", [0.98, 0.9])
def test_corner_radii_match_loop_implementation(max_dev_thresh, circle_fit_thresh):
    for name, mask in _masks():
        keypoints, r_max, r_max_pos = _corner_inputs(mask, max_dev_thresh)
        expected = _loop_corner_radii(keypoints, r_max, r_max_pos, circle_fit_thresh)
        actual = compute_corner_radii(keypoints, r_max, r_max_pos, circle_fit_thresh)

        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(actual, expected, rtol=1e-9, err_msg=name)


def test_prefix_circle_fits_match_least_squares():
    rng = np.random.default_rng(0)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, 40))
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 50.0 + [120.0, 80.0]
    points = (points + rng.normal(0.0, 0.5, points.shape)).astype(np.float32)
    sizes = np.arange(3, len(points) + 1)

    radius, center = _fit_circles(points, sizes)

    for i, size in enumerate(sizes):
        expected_radius, expected_center = _fit_circle(points[:size])
        np.testing.assert_allclose(radius[i], expected_radius, rtol=1e-9)
        np.testing.assert_allclose(center[i], expected_center, rtol=1e-9)


def test_collinear_prefix_falls_back_to_least_squares():
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    sizes = np.array([3, 4])

    radius, center = _fit_circles(points, sizes)

    for i, size in enumerate(sizes):
        expected_radius, expected_center = _fit_circle(points[:size])
        np.testing.assert_allclose(radius[i], expected_radius)
        np.testing.assert_allclose(center[i], expected_center)
