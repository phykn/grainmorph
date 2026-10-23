import math

import numpy as np


def discretize_boundary(boundary: np.ndarray, max_dev_thresh: float) -> np.ndarray:
    num_pts = len(boundary)

    segment_start = 0
    segment_end = num_pts
    keypoints = [boundary[0]]

    boundary = np.vstack([boundary, boundary[0]])
    total_length = len(boundary)

    while segment_start < num_pts:
        max_dev, position_index = max_line_deviation(
            boundary[segment_start:segment_end]
        )

        while max_dev > max_dev_thresh:
            new_end = position_index + segment_start

            if new_end <= segment_start:
                segment_end = segment_start + 2
                break

            segment_end = new_end
            max_dev, position_index = max_line_deviation(
                boundary[segment_start:segment_end]
            )

        if segment_end != total_length or segment_start == 0:
            idx = min(segment_end - 1, total_length - 1)
            keypoints.append(boundary[idx])

        segment_start = segment_end
        segment_end = total_length

    return np.array(keypoints)


def max_line_deviation(points: np.ndarray) -> tuple[float, int]:
    if len(points) <= 1:
        return (0.0, 0)

    x = points[:, 0]
    y = points[:, 1]
    x0, y0 = points[0].tolist()
    x1, y1 = points[-1].tolist()

    dx = x1 - x0
    dy = y0 - y1
    dist_end_sq = dx * dx + dy * dy

    if dist_end_sq < 1e-12:
        dist_sq = np.square(x - x0) + np.square(y - y0)
        idx = int(dist_sq.argmax())

        return (math.sqrt(float(dist_sq[idx])), idx)

    c = y1 * x0 - y0 * x1
    dist_num = np.abs(dy * x + dx * y + c)
    idx = int(dist_num.argmax())

    return (float(dist_num[idx]) / math.sqrt(dist_end_sq), idx)
