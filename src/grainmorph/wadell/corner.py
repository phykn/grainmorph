import numpy as np


def compute_corner_radii(keypoints, r_max, r_max_pos, circle_fit_thresh):
    convex = _select_convex_points(keypoints).astype(np.float32)
    keypoints = keypoints.astype(np.float32)
    num = len(convex)
    dist_cumsum = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(convex - r_max_pos, axis=1))]
    )

    radii = []
    start = 0
    while start + 3 <= num:
        ends = np.arange(start + 3, num + 1)
        sizes = ends - start
        radius, center = _fit_circles(convex[start:], sizes)

        dist_center = np.linalg.norm(center - r_max_pos, axis=1)
        dist_mean = (dist_cumsum[ends] - dist_cumsum[start]) / sizes
        idx = np.flatnonzero(
            (radius > 1e-06) & (dist_center < dist_mean) & (radius < r_max)
        )

        dist_all = np.linalg.norm(keypoints[None] - center[idx, None], axis=2)
        hits = idx[np.all(dist_all / radius[idx, None] > circle_fit_thresh, axis=1)]
        if len(hits) > 0:
            radii.append(float(radius[hits[-1]]))
            start = int(ends[hits[-1]])
        else:
            start += 1

    return np.array(radii)


def _select_convex_points(keypoints):
    wrapped = np.vstack([keypoints[-1], keypoints, keypoints[0]])
    vec1 = wrapped[1:-1] - wrapped[:-2]
    vec2 = wrapped[2:] - wrapped[1:-1]
    angle = np.arctan2(
        vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0],
        vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1],
    )

    x, y = keypoints.T
    signed_area = (
        np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0]
    )
    if signed_area < 0.0:
        angle = -angle

    concave = np.flatnonzero(angle < 0.0)
    first = concave[0] if len(concave) > 0 else int(np.argmax(angle))
    return np.roll(keypoints, -first, axis=0)[np.roll(angle, -first) > 0.0]


def _fit_circles(points, sizes):
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


def _fit_circle(points):
    points = points.astype(np.float64)
    low = points.min(axis=0)
    high = points.max(axis=0)
    scale = 0.5 * (high - low).max()
    if scale < 1e-12:
        return 0.0, points[0]

    offset = (high + low) * 0.5
    normed = (points - offset) / scale
    design = np.column_stack([normed, np.ones(len(normed))])
    solution, *_ = np.linalg.lstsq(
        design, np.square(normed).sum(axis=1), rcond=np.finfo(float).eps
    )

    center = 0.5 * solution[:2]
    radius = np.sqrt(solution[2] + np.square(center).sum())
    return radius * scale, center * scale + offset
