import cv2
import numpy as np

from .wadell.corner import compute_corner_radii
from .wadell.discretize import discretize_boundary
from .wadell.smoothing import smooth_boundary


def compute_aspect_ratio(mask, eps):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = contours[0].reshape(-1, 2).astype(np.float32)

    if len(points) < 5:
        rect = cv2.minAreaRect(points)
        (x, y), (w, h), angle = rect
        bbox = cv2.boxPoints(rect).tolist()
    else:
        (x, y), (w, h), angle = cv2.fitEllipse(points)
        theta = np.deg2rad(angle)
        axes = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        projected = points @ axes
        x_min, y_min = projected.min(axis=0)
        x_max, y_max = projected.max(axis=0)
        corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        bbox = (np.array(corners) @ axes.T).tolist()

    if min(w, h) < eps:
        return None

    return {
        "val": max(w, h) / min(w, h),
        "ellipse": {
            "major": max(w, h),
            "minor": min(w, h),
            "x": x,
            "y": y,
            "angle": angle,
            "w": w,
            "h": h,
            "bbox": bbox,
        },
    }


def compute_roundness(
    mask,
    max_dev_thresh=0.3,
    circle_fit_thresh=0.98,
    alpha_ratio=0.05,
    beta_ratio=0.001,
):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = contours[0].reshape(-1, 2).astype(np.float64)
    if len(boundary) < 3:
        return None

    smoothed = smooth_boundary(boundary, alpha_ratio, beta_ratio)
    keypoints = discretize_boundary(smoothed, max_dev_thresh)
    if len(keypoints) < 3:
        return None

    inscribed = find_inscribed_circle(mask)
    radii = compute_corner_radii(
        keypoints,
        inscribed["r"],
        np.array([inscribed["x"], inscribed["y"]]),
        circle_fit_thresh,
    )
    if len(radii) == 0:
        return None

    return {"val": float(np.clip(np.mean(radii) / inscribed["r"], 0.0, 1.0))}


def compute_circularity(mask, eps):
    # Four-direction Crofton weights indexed by 2x2 pixel pattern, matching
    # skimage.measure.perimeter_crofton(image, directions=4).
    weights = np.array(
        [
            0,
            np.pi / 4 * (1 + 1 / np.sqrt(2)),
            np.pi / (4 * np.sqrt(2)),
            np.pi / (2 * np.sqrt(2)),
            0,
            np.pi / 4 * (1 + 1 / np.sqrt(2)),
            0,
            np.pi / (4 * np.sqrt(2)),
            np.pi / 4,
            np.pi / 2,
            np.pi / (4 * np.sqrt(2)),
            np.pi / (4 * np.sqrt(2)),
            np.pi / 4,
            np.pi / 2,
            0,
            0,
        ]
    )
    padded = np.pad(mask, 1)
    patterns = (
        padded[1:, 1:]
        + 2 * padded[:-1, 1:]
        + 4 * padded[1:, :-1]
        + 8 * padded[:-1, :-1]
    )
    perimeter = float(weights @ np.bincount(patterns.ravel(), minlength=16))
    if perimeter < eps:
        return None

    area = np.count_nonzero(mask)
    return {"val": float(np.clip(4.0 * np.pi * area / perimeter**2, 0.0, 1.0))}


def compute_sphericity(mask, eps):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), r = cv2.minEnclosingCircle(cv2.convexHull(contours[0]))
    if r < eps:
        return None

    inscribed = find_inscribed_circle(mask)
    return {
        "val": float(np.clip(inscribed["r"] / r, 0.0, 1.0)),
        "inscribed": inscribed,
        "enclosing": {"x": x, "y": y, "r": r},
    }


def find_inscribed_circle(mask):
    # Padding makes the image border count as background.
    dist = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return {"x": float(x - 1), "y": float(y - 1), "r": float(dist[y, x])}
