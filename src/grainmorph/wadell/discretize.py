import math

import numpy as np


def discretize_boundary(boundary, max_dev_thresh):
    num = len(boundary)
    # Separate columns and plain floats keep each of the many short calls cheap.
    x = np.append(boundary[:, 0], boundary[0, 0])
    y = np.append(boundary[:, 1], boundary[0, 1])
    xs = x.tolist()
    ys = y.tolist()
    keep = [0]

    start, end = 0, num
    while start < num:
        dev, idx = find_max_deviation(
            x[start : end + 1],
            y[start : end + 1],
            xs[start],
            ys[start],
            xs[end],
            ys[end],
        )
        # A peak at either end would not shrink the segment.
        while dev > max_dev_thresh and 0 < idx < end - start:
            end = start + idx
            dev, idx = find_max_deviation(
                x[start : end + 1],
                y[start : end + 1],
                xs[start],
                ys[start],
                xs[end],
                ys[end],
            )

        if end < num:
            keep.append(end)
        start, end = end, num

    return boundary[keep]


def find_max_deviation(x, y, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y0 - y1
    chord_sq = dx * dx + dy * dy

    if chord_sq < 1e-12:
        dist_sq = np.square(x - x0) + np.square(y - y0)
        idx = int(dist_sq.argmax())
        return math.sqrt(float(dist_sq[idx])), idx

    dist = dy * x
    dist += dx * y
    dist += y1 * x0 - y0 * x1
    np.abs(dist, out=dist)
    idx = int(dist.argmax())
    return float(dist[idx]) / math.sqrt(chord_sq), idx
