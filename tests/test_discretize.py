import numpy as np
import pytest

from grainmorph.wadell.discretize import discretize_boundary, find_max_deviation


def test_discretize_keeps_exactly_the_corners_of_a_square():
    side = np.arange(10.0)
    zeros = np.zeros(10)
    boundary = np.concatenate(
        [
            np.stack([side, zeros], 1),
            np.stack([zeros + 10, side], 1),
            np.stack([10 - side, zeros + 10], 1),
            np.stack([zeros, 10 - side], 1),
        ]
    )

    keypoints = discretize_boundary(boundary, 0.3)

    assert keypoints.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]


@pytest.mark.parametrize("angle", [0, 30, 90, 135])
@pytest.mark.parametrize("offset", [(0, 0), (13, -7)])
def test_max_deviation_is_invariant_to_rigid_transforms(angle, offset):
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 0.0]])
    theta = np.deg2rad(angle)
    axes = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    x, y = (points @ axes.T + offset).T

    deviation, idx = find_max_deviation(x, y, x[0], y[0], x[-1], y[-1])

    assert deviation == pytest.approx(2.0)
    assert idx == 1


def test_max_deviation_with_coincident_endpoints():
    x = np.array([1.0, 4.0, 1.0])
    y = np.array([2.0, 6.0, 2.0])

    assert find_max_deviation(x, y, 1.0, 2.0, 1.0, 2.0) == (5.0, 1)
