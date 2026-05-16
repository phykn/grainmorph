import numpy as np


def discretize_boundary(boundary: np.ndarray, max_dev_thresh: float) -> np.ndarray:
    x = boundary[:, 0]
    y = boundary[:, 1]
    num_pts = len(x)

    segment_start = 0
    segment_end = num_pts
    keypoints = [[x[segment_start], y[segment_start]]]

    x = np.append(x, x[0])
    y = np.append(y, y[0])
    total_length = len(x)

    while segment_start < num_pts:
        max_dev, position_index = max_line_deviation(
            x[segment_start:segment_end], y[segment_start:segment_end]
        )

        while max_dev > max_dev_thresh:
            new_end = position_index + segment_start

            if new_end <= segment_start:
                segment_end = segment_start + 2
                break

            segment_end = new_end
            max_dev, position_index = max_line_deviation(
                x[segment_start:segment_end], y[segment_start:segment_end]
            )

        if segment_end != total_length or segment_start == 0:
            idx = min(segment_end - 1, total_length - 1)
            keypoints.append([x[idx], y[idx]])

        segment_start = segment_end
        segment_end = total_length

    return np.array(keypoints)


def classify_concave_convex(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrapped_kpts = np.vstack([keypoints[-1], keypoints, keypoints[0]])

    vec1 = wrapped_kpts[1:-1] - wrapped_kpts[:-2]
    vec2 = wrapped_kpts[2:] - wrapped_kpts[1:-1]

    angle = np.arctan2(
        vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0],
        vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1],
    )

    area = signed_area(keypoints)
    orientation = 1.0 if area >= 0.0 else -1.0
    oriented = angle * orientation

    concave_locs = np.where(oriented < 0.0)[0]
    roll_idx = concave_locs[0] if len(concave_locs) > 0 else int(np.argmax(oriented))

    keypoints = np.roll(keypoints, -roll_idx, axis=0)
    oriented = np.roll(oriented, -roll_idx)

    return (keypoints[oriented < 0.0], keypoints[oriented > 0.0])


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]

    area = np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0]

    return 0.5 * float(area)


def max_line_deviation(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    if len(x) <= 1:
        return (0.0, 0)

    x0, x1 = (x[0], x[-1])
    y0, y1 = (y[0], y[-1])

    dx = x1 - x0
    dy = y0 - y1
    dist_end_sq = dx * dx + dy * dy

    if dist_end_sq < 1e-12:
        dx_arr = x - x0
        dy_arr = y - y0
        dist_sq = dx_arr * dx_arr + dy_arr * dy_arr
        idx = int(np.argmax(dist_sq))

        return (float(np.sqrt(dist_sq[idx])), idx)

    c = y1 * x0 - y0 * x1
    dist_num = np.abs(dy * x + dx * y + c)
    idx = int(np.argmax(dist_num))

    return (float(dist_num[idx] / np.sqrt(dist_end_sq)), idx)
