import numpy as np
import pytest

from grainmorph.wadell.discretize import max_line_deviation


@pytest.mark.parametrize("angle", [0, 30, 90, 135])
@pytest.mark.parametrize("offset", [(0, 0), (13, -7)])
def test_line_deviation_is_invariant_to_rigid_transforms(angle, offset):
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 0.0]])
    theta = np.deg2rad(angle)
    axes = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = points @ axes.T + offset

    deviation, idx = max_line_deviation(points)

    assert deviation == pytest.approx(2.0)
    assert idx == 1


def test_line_deviation_with_coincident_endpoints():
    points = np.array([[1.0, 2.0], [4.0, 6.0], [1.0, 2.0]])

    assert max_line_deviation(points) == (5.0, 1)
