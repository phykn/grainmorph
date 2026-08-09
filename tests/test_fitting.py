import cv2
import numpy as np

import partimorph as pm


def test_aspect_ratio_payload_uses_fitted_ellipse():
    mask = np.zeros((220, 220), dtype=np.uint8)
    vertices = np.array([[20, 180], [100, 20], [180, 180]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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
