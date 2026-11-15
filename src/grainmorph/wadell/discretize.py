import math

import numpy as np


def discretize_boundary(boundary, max_dev_thresh):
    num = len(boundary)
    closed = np.vstack([boundary, boundary[0]])
    keypoints = [boundary[0]]

    start, end = 0, num
    while start < num:
        dev, idx = find_max_deviation(closed[start:end])
        while dev > max_dev_thresh:
            if idx == 0:
                end = start + 2
                break
            end = start + idx
            dev, idx = find_max_deviation(closed[start:end])

        if end <= num or start == 0:
            keypoints.append(closed[end - 1])
        start, end = end, num + 1

    return np.array(keypoints)


def find_max_deviation(points):
    if len(points) <= 1:
        return 0.0, 0

    x, y = points.T
    x0, y0 = points[0].tolist()
    x1, y1 = points[-1].tolist()
    dx = x1 - x0
    dy = y0 - y1
    chord_sq = dx * dx + dy * dy

    if chord_sq < 1e-12:
        dist_sq = np.square(x - x0) + np.square(y - y0)
        idx = int(dist_sq.argmax())
        return math.sqrt(float(dist_sq[idx])), idx

    dist = np.abs(dy * x + dx * y + (y1 * x0 - y0 * x1))
    idx = int(dist.argmax())
    return float(dist[idx]) / math.sqrt(chord_sq), idx
