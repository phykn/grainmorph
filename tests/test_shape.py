import numpy as np
import pytest

import grainmorph as gm


def test_triangle_vertices_use_cartesian_order():
    vertices = ((30, 2), (40, 2), (30, 8))

    mask = gm.utils.create_triangle_mask((20, 60), *vertices)

    assert mask.dtype == bool
    assert all(mask[y, x] for x, y in vertices)
    y, x = np.nonzero(mask)
    assert (y.min(), y.max(), x.min(), x.max()) == (2, 8, 30, 40)


def test_centered_shapes_use_image_order():
    shape = (120, 200)
    center = (50, 130)
    masks = [
        gm.utils.create_circle_mask(shape, center, 30),
        gm.utils.create_ellipse_mask(shape, center, 40, 25),
        gm.utils.create_pentagon_mask(shape, center, 40),
        gm.utils.create_star_mask(shape, center, 40, 15, num_points=6),
        gm.utils.create_square_mask(shape, (30, 110), 41),
        gm.utils.create_rectangle_mask(shape, (40, 100), (61, 161)),
    ]

    for mask in masks:
        assert mask.shape == shape and mask.dtype == bool
        y, x = np.nonzero(mask)
        np.testing.assert_allclose([y.mean(), x.mean()], center, atol=0.2)


def test_shapes_reject_invalid_sizes():
    shape = (16, 16)
    with pytest.raises(ValueError, match="finite"):
        gm.utils.create_circle_mask(shape, (8, 8), np.nan)
    with pytest.raises(ValueError, match="finite"):
        gm.utils.create_ellipse_mask(shape, (8, 8), np.inf, 2)
    with pytest.raises(ValueError, match="finite"):
        gm.utils.create_pentagon_mask(shape, (8, 8), np.nan)
    with pytest.raises(ValueError, match="finite"):
        gm.utils.create_star_mask(shape, (8, 8), 5, np.inf)
    with pytest.raises(ValueError, match="num_points"):
        gm.utils.create_star_mask(shape, (8, 8), 5, 2, num_points=1)
    with pytest.raises(ValueError, match="size"):
        gm.utils.create_square_mask(shape, (2, 2), 0)
    with pytest.raises(ValueError, match="within mask bounds"):
        gm.utils.create_rectangle_mask(shape, (2, 2), (20, 8))
    with pytest.raises(ValueError, match="above-left"):
        gm.utils.create_rectangle_mask(shape, (8, 2), (2, 8))
