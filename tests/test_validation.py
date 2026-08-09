import numpy as np
import pytest

from partimorph.validation import to_binary


def test_to_binary_validation():
    bool_mask = np.zeros((10, 10), dtype=bool)
    result = to_binary(bool_mask)
    assert result.dtype == np.uint8
    assert np.all(np.isin(result, [0, 1]))

    int_mask = np.array([[0, 1], [1, 0]], dtype=np.int32)
    result = to_binary(int_mask)
    assert result.dtype == np.uint8

    with pytest.raises(ValueError, match="binary with values in {0, 1}"):
        to_binary(np.array([[0, 2], [1, 0]]))

    with pytest.raises(ValueError, match="mask must be 2D"):
        to_binary(np.zeros((10, 10, 10)))
