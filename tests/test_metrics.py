import cv2
import numpy as np
import pytest

from grainmorph.metrics import (
    compute_aspect_ratio,
    compute_circularity,
    compute_roundness,
    compute_sphericity,
)


def _draw_ellipse(angle):
    mask = np.zeros((160, 180), dtype=np.uint8)
    cv2.ellipse(mask, (90, 80), (50, 18), angle, 0, 360, 1, -1)
    return mask


def _draw_disk(radius):
    y, x = np.ogrid[:50, :50]
    return ((x - 25) ** 2 + (y - 25) ** 2 <= radius**2).astype(np.uint8)


@pytest.mark.parametrize("angle", [0, 30, 75])
def test_aspect_ratio_uses_fitted_ellipse(angle):
    mask = _draw_ellipse(angle)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), (w, h), fitted_angle = cv2.fitEllipse(contours[0].astype(np.float32))

    result = compute_aspect_ratio(mask, 0.001)

    ellipse = result["ellipse"]
    assert result["val"] == pytest.approx(max(w, h) / min(w, h))
    assert result["val"] == ellipse["major"] / ellipse["minor"]
    np.testing.assert_allclose(
        [ellipse["x"], ellipse["y"], ellipse["w"], ellipse["h"], ellipse["angle"]],
        [x, y, w, h, fitted_angle],
    )
    theta = np.deg2rad(ellipse["angle"])
    axes = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rows, cols = np.nonzero(mask)
    projected = np.column_stack([cols, rows]) @ axes
    box = np.asarray(ellipse["bbox"]) @ axes
    np.testing.assert_allclose(box.min(axis=0), projected.min(axis=0))
    np.testing.assert_allclose(box.max(axis=0), projected.max(axis=0))


def test_aspect_ratio_of_rectangle_uses_its_corners():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[5:25, 10:50] = 1

    ellipse = compute_aspect_ratio(mask, 0.001)["ellipse"]

    np.testing.assert_allclose([ellipse["x"], ellipse["y"]], [29.5, 14.5])
    np.testing.assert_allclose([ellipse["major"], ellipse["minor"]], [39, 19])
    np.testing.assert_allclose(np.min(ellipse["bbox"], axis=0), [10, 5])
    np.testing.assert_allclose(np.max(ellipse["bbox"], axis=0), [49, 24])


def test_roundness_separates_round_from_angular():
    star = np.zeros((200, 200), dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = np.where(np.arange(10) % 2, 35, 90)
    points = np.stack([100 + radii * np.cos(angles), 100 + radii * np.sin(angles)], 1)
    cv2.fillPoly(star, [points.astype(np.int32)], 1)

    assert compute_roundness(_draw_disk(20))["val"] > 0.95
    assert compute_roundness(star)["val"] < 0.3
    assert compute_roundness(np.ones((1, 2), dtype=np.uint8)) is None


@pytest.mark.parametrize(
    ("mask", "perimeter"),
    [
        (np.ones((10, 10), dtype=np.uint8), 36.811657224201205),
        (_draw_disk(20), 127.71373126734748),
        (np.eye(6, dtype=np.uint8), 13.31230053165795),
    ],
)
def test_circularity_uses_scikit_image_crofton_perimeter(mask, perimeter):
    # Perimeters come from skimage.measure.perimeter_crofton(mask, 4).
    expected = 4 * np.pi * np.count_nonzero(mask) / perimeter**2

    assert compute_circularity(mask, 0.001)["val"] == pytest.approx(expected, rel=1e-12)
    assert compute_circularity(mask, 1000.0) is None


def test_sphericity_is_inscribed_over_enclosing_radius():
    mask = np.ones((21, 41), dtype=np.uint8)

    result = compute_sphericity(mask, 0.001)

    assert result["inscribed"]["r"] == 11.0
    assert result["inscribed"]["y"] == 10.0
    np.testing.assert_allclose(
        [result["enclosing"][key] for key in "xyr"],
        [20, 10, np.hypot(20, 10)],
        rtol=1e-4,
    )
    assert result["val"] == pytest.approx(11 / np.hypot(20, 10), rel=1e-4)
    assert compute_sphericity(np.ones((1, 1), dtype=np.uint8), 0.001) is None
