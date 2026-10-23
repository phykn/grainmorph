import numpy as np

from partimorph.mask_ops import prepare_mask


def test_prepare_mask_separates_diagonal_components():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:10, 2:10] = 1
    mask[10:14, 10:14] = 1

    prepared = prepare_mask(mask, target_dim=384)

    np.testing.assert_array_equal(prepared.mask, np.ones((8, 8), dtype=np.uint8))
    assert (prepared.offset_x, prepared.offset_y) == (2.0, 2.0)


def test_prepare_mask_fills_holes_closed_by_four_connectivity():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[1:8, 1:8] = 1
    mask[3:6, 3:6] = 0
    mask[1, 1] = 0
    mask[2, 2] = 0

    prepared = prepare_mask(mask, target_dim=384)

    expected = np.ones((7, 7), dtype=np.uint8)
    expected[0, 0] = 0
    np.testing.assert_array_equal(prepared.mask, expected)


def test_prepare_mask_keeps_notches_open_to_the_border():
    mask = np.ones((7, 7), dtype=np.uint8)
    mask[0:4, 3] = 0

    prepared = prepare_mask(mask, target_dim=384)

    np.testing.assert_array_equal(prepared.mask, mask)
