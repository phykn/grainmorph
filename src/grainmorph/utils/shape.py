import cv2
import numpy as np


def create_circle_mask(shape, center, radius):
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and >= 0.")

    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def create_ellipse_mask(shape, center, radius_x, radius_y):
    if (
        not (np.isfinite(radius_x) and np.isfinite(radius_y))
        or min(radius_x, radius_y) <= 0
    ):
        raise ValueError("radius_x and radius_y must be finite and > 0.")

    y, x = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    return ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2 <= 1.0


def create_rectangle_mask(shape, top_left, bottom_right):
    y1, x1 = top_left
    y2, x2 = bottom_right
    if y1 >= y2 or x1 >= x2:
        raise ValueError("top_left must be strictly above-left of bottom_right.")
    if y1 < 0 or x1 < 0 or y2 > shape[0] or x2 > shape[1]:
        raise ValueError("rectangle coordinates must be within mask bounds.")

    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def create_square_mask(shape, top_left, size):
    if size <= 0:
        raise ValueError("size must be > 0.")

    return create_rectangle_mask(
        shape, top_left, (top_left[0] + size, top_left[1] + size)
    )


def create_triangle_mask(shape, v1, v2, v3):
    return fill_polygon(shape, np.array([v1, v2, v3]))


def create_pentagon_mask(shape, center, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and > 0.")

    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) - np.pi / 2
    return fill_polygon(shape, compute_polar_vertices(center, radius, angles))


def create_star_mask(shape, center, outer_radius, inner_radius, num_points=5):
    if not np.isfinite(outer_radius) or outer_radius <= 0:
        raise ValueError("outer_radius must be finite and > 0.")
    if not np.isfinite(inner_radius) or inner_radius <= 0:
        raise ValueError("inner_radius must be finite and > 0.")
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")

    angles = np.linspace(0, 2 * np.pi, 2 * num_points, endpoint=False) - np.pi / 2
    radii = np.full(len(angles), float(outer_radius))
    radii[1::2] = inner_radius
    return fill_polygon(shape, compute_polar_vertices(center, radii, angles))


def compute_polar_vertices(center, radii, angles):
    cy, cx = center
    return np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)


def fill_polygon(shape, vertices):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 1)
    return mask.astype(bool)
