import cv2
import numpy as np

from .mask_ops import crop_mask, get_contours
from .schema import Bbox, CircleData, EllipseData, Mask, Points


def find_inscribed_circle(mask: Mask) -> CircleData | None:
    cropped_mask, pad_x0, pad_y0 = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return None

    distance_transform = cv2.distanceTransform(
        cropped_mask.astype(np.uint8, copy=False), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    max_idx = np.argmax(distance_transform)
    crop_center_y, crop_center_x = np.unravel_index(max_idx, distance_transform.shape)

    center_x = float(int(crop_center_x) + pad_x0)
    center_y = float(int(crop_center_y) + pad_y0)
    radius = float(distance_transform[crop_center_y, crop_center_x])

    return {"x": center_x, "y": center_y, "r": radius}


def find_enclosing_circle(mask: Mask) -> CircleData | None:
    contours = get_contours(mask)
    if not contours:
        return None

    points = np.concatenate(contours)
    hull = cv2.convexHull(points)

    (center_x, center_y), radius = cv2.minEnclosingCircle(hull)

    return {"x": float(center_x), "y": float(center_y), "r": float(radius)}


def fit_ellipse(mask: Mask) -> EllipseData | None:
    contours = get_contours(mask)
    if not contours:
        return None

    points: Points = np.concatenate(contours).reshape(-1, 2).astype(np.float32)
    if len(points) < 5:
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), angle = rect

        if width <= 1e-06 or height <= 1e-06:
            return None

        bbox = cv2.boxPoints(rect).tolist()
    else:
        (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)
        bbox = _oriented_bbox_from_angle(points, angle)

    return {
        "major": float(max(width, height)),
        "minor": float(min(width, height)),
        "x": float(center_x),
        "y": float(center_y),
        "angle": float(angle),
        "w": float(width),
        "h": float(height),
        "bbox": bbox,
    }


def _oriented_bbox_from_angle(points: Points, angle: float) -> Bbox:
    radians = np.deg2rad(angle)
    cos_a, sin_a = np.cos(radians), np.sin(radians)
    axes = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    projected = points @ axes
    min_x, min_y = projected.min(axis=0)
    max_x, max_y = projected.max(axis=0)
    corners = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    return (corners @ axes.T).tolist()
