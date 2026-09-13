import cv2
import numpy as np
import pytest

import partimorph as pm
from partimorph.fitting import find_inscribed_circle, fit_ellipse


def test_aspect_ratio_payload_uses_fitted_ellipse():
    mask = np.zeros((220, 220), dtype=np.uint8)
    vertices = np.array([[20, 180], [100, 20], [180, 180]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.concatenate(contours).astype(np.float32)
    (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)

    results = pm.analyze_mask(
        mask,
        use_roundness=False,
        use_circularity=False,
        use_sphericity=False,
    )

    assert results is not None
    assert results["aspect_ratio"] is not None
    ellipse = results["aspect_ratio"]["ellipse"]
    np.testing.assert_allclose(
        [ellipse["x"], ellipse["y"], ellipse["w"], ellipse["h"], ellipse["angle"]],
        [center_x, center_y, width, height, angle],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        results["aspect_ratio"]["val"],
        max(width, height) / min(width, height),
    )
    assert len(ellipse["bbox"]) == 4


@pytest.mark.parametrize("angle", [0, 30, 75])
def test_ellipse_bbox_encloses_contour_in_fitted_axes(angle):
    mask = np.zeros((160, 180), dtype=np.uint8)
    cv2.ellipse(mask, (90, 80), (50, 18), angle, 0, 360, 1, -1)

    ellipse = fit_ellipse(mask)
    box = np.asarray(ellipse["bbox"])
    assert box.shape == (4, 2)
    theta = np.deg2rad(ellipse["angle"])
    axes = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    y, x = np.nonzero(mask)
    projected = np.column_stack([x, y]) @ axes
    projected_box = box @ axes
    np.testing.assert_allclose(projected_box.min(axis=0), projected.min(axis=0))
    np.testing.assert_allclose(projected_box.max(axis=0), projected.max(axis=0))


def test_ellipse_fallback_preserves_rectangle_geometry():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[5:25, 10:50] = 1

    ellipse = fit_ellipse(mask)

    np.testing.assert_allclose([ellipse["x"], ellipse["y"]], [29.5, 14.5])
    np.testing.assert_allclose([ellipse["major"], ellipse["minor"]], [39, 19])
    np.testing.assert_allclose(np.min(ellipse["bbox"], axis=0), [10, 5])
    np.testing.assert_allclose(np.max(ellipse["bbox"], axis=0), [49, 24])


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int32])
def test_inscribed_circle_preserves_binary_mask_dtypes(dtype):
    mask = np.zeros((20, 30), dtype=dtype)
    mask[5:14, 10:19] = 1

    assert find_inscribed_circle(mask) == {"x": 14.0, "y": 9.0, "r": 5.0}
