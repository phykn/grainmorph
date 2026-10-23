import numpy as np


def compute_corner_radii(
    keypoints: np.ndarray,
    r_max: float,
    r_max_pos: np.ndarray,
    circle_fit_thresh: float,
) -> np.ndarray:
    convex_points = _select_convex_points(keypoints).astype(np.float32)
    keypoints = keypoints.astype(np.float32)
    num_points = len(convex_points)

    dist_boundary = np.linalg.norm(convex_points - r_max_pos, axis=1)
    dist_cumsum = np.concatenate([[0.0], np.cumsum(dist_boundary)])

    radii: list[float] = []
    start = 0

    while True:
        ends = np.arange(start + 3, num_points + 1)
        if len(ends) == 0:
            break

        sizes = ends - start
        radius, center = _fit_circles(convex_points[start:], sizes)

        dist_center = np.linalg.norm(center - r_max_pos, axis=1)
        dist_mean = (dist_cumsum[ends] - dist_cumsum[start]) / sizes

        with np.errstate(invalid="ignore"):
            valid = (
                np.isfinite(radius)
                & (radius > 1e-06)
                & (dist_center < dist_mean)
                & (radius < r_max)
            )
        idx = np.flatnonzero(valid)

        if len(idx) > 0:
            dist_all = np.linalg.norm(keypoints[None] - center[idx, None], axis=2)
            fits = np.all(dist_all / radius[idx, None] > circle_fit_thresh, axis=1)
            hits = idx[fits]

            if len(hits) > 0:
                best = hits[-1]
                radii.append(float(radius[best]))
                start = int(ends[best])
                continue

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


def _fit_circles(
    points: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Moments are taken relative to the first point so the centered sums stay
    # accurate for short prefixes.
    origin = points[0].astype(np.float64)
    x, y = (points.astype(np.float64) - origin).T
    sums = np.cumsum(
        np.stack([x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3]),
        axis=1,
    )[:, sizes - 1]
    sx, sy, sxx, sxy, syy, sxxx, sxxy, sxyy, syyy = sums

    n = sizes.astype(np.float64)
    mx = sx / n
    my = sy / n

    suu = sxx - n * mx * mx
    suv = sxy - n * mx * my
    svv = syy - n * my * my
    suuu = sxxx - 3 * mx * sxx + 2 * n * mx**3
    svvv = syyy - 3 * my * syy + 2 * n * my**3
    suuv = sxxy - my * sxx - 2 * mx * sxy + 2 * n * mx * mx * my
    suvv = sxyy - mx * syy - 2 * my * sxy + 2 * n * mx * my * my

    det = suu * svv - suv * suv
    # Nearly collinear prefixes make the normal equations ill-conditioned.
    singular = det <= 1e-06 * np.square(suu + svv)
    det = np.where(singular, 1.0, det)

    rhs_u = 0.5 * (suuu + suvv)
    rhs_v = 0.5 * (suuv + svvv)
    uc = (svv * rhs_u - suv * rhs_v) / det
    vc = (suu * rhs_v - suv * rhs_u) / det

    radius = np.sqrt(uc * uc + vc * vc + (suu + svv) / n)
    center = np.stack([uc + mx, vc + my], axis=1) + origin

    for i in np.flatnonzero(singular):
        radius[i], center[i] = _fit_circle(points[: sizes[i]])

    return radius, center


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
    y, *_ = np.linalg.lstsq(B, d, rcond=np.finfo(float).eps)

    norm_center = 0.5 * y[:2]
    norm_radius = float(np.sqrt(y[2] + np.square(norm_center).sum()))
    center = norm_center * scale + offset

    if not np.all(np.isfinite(center)):
        return (0.0, points[0].copy())

    radius = norm_radius * scale

    if not np.isfinite(radius):
        return (0.0, center)

    return (radius, center)
