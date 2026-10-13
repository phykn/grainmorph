import numpy as np
from scipy.linalg import lstsq


def compute_corner_radii(
    keypoints: np.ndarray,
    r_max: float,
    r_max_pos: np.ndarray,
    circle_fit_thresh: float,
) -> np.ndarray:
    convex_points = _select_convex_points(keypoints).astype(np.float32)
    keypoints = keypoints.astype(np.float32)
    num_points = len(convex_points)

    radii: list[float] = []
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


def _select_convex_points(keypoints: np.ndarray) -> np.ndarray:
    wrapped = np.vstack([keypoints[-1], keypoints, keypoints[0]])
    vec1 = wrapped[1:-1] - wrapped[:-2]
    vec2 = wrapped[2:] - wrapped[1:-1]

    angle = np.arctan2(
        vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0],
        vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1],
    )
    orientation = 1.0 if _signed_area(keypoints) >= 0.0 else -1.0
    oriented = angle * orientation

    concave_locs = np.where(oriented < 0.0)[0]
    idx = concave_locs[0] if len(concave_locs) > 0 else int(np.argmax(oriented))
    keypoints = np.roll(keypoints, -idx, axis=0)
    oriented = np.roll(oriented, -idx)
    return keypoints[oriented > 0.0]


def _signed_area(points: np.ndarray) -> float:
    x, y = points.T
    area = np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0]
    return 0.5 * float(area)


def _fit_circle(points: np.ndarray) -> tuple[float, np.ndarray]:
    points = points.reshape(-1, 2).copy()
    num_pts = points.shape[0]

    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    x_span = x_max - x_min
    y_span = y_max - y_min
    scale = 0.5 * max(x_span, y_span)

    if scale < 1e-12:
        return (0.0, points[0].copy())

    offset = np.array([(x_max + x_min) * 0.5, (y_max + y_min) * 0.5], dtype=np.float64)
    points_norm = (points - offset) / scale

    B = np.empty((num_pts, 3), dtype=points_norm.dtype)
    B[:, :2] = points_norm
    B[:, 2] = 1.0

    d = np.square(points_norm).sum(axis=1)
    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)

    norm_center = 0.5 * y[:2]
    norm_radius = float(np.sqrt(y[2] + np.square(norm_center).sum()))
    center = norm_center * scale + offset

    if not np.all(np.isfinite(center)):
        return (0.0, points[0].copy())

    radius = norm_radius * scale

    if not np.isfinite(radius):
        return (0.0, center)

    return (radius, center)
