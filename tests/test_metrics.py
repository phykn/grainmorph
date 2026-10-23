import numpy as np
import pytest

import grainmorph as gm
from grainmorph.metrics import _crofton_perimeter
from grainmorph.wadell import compute_roundness


@pytest.mark.parametrize(
    "params",
    [{}, {"max_dev_thresh": 0.5, "alpha_ratio": 0.025, "beta_ratio": 0.0005}],
)
def test_analysis_roundness_uses_wadell_options(params):
    mask = gm.utils.create_circle_mask((100, 100), (50, 50), 30)
    expected = compute_roundness(mask, **params)

    results = gm.analyze_mask(
        mask,
        use_aspect_ratio=False,
        use_circularity=False,
        use_sphericity=False,
        roundness_params=params,
    )

    assert expected is not None
    np.testing.assert_allclose(results["roundness"]["val"], np.clip(expected, 0.0, 1.0))


def _disk(radius):
    y, x = np.ogrid[:50, :50]
    return ((x - 25) ** 2 + (y - 25) ** 2 <= radius**2).astype(np.uint8)


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (np.pad(np.ones((10, 10), dtype=np.uint8), 1), 36.811657224201205),
        (_disk(20), 127.71373126734748),
        (np.eye(6, dtype=np.uint8), 13.31230053165795),
        (np.ones((1, 1), dtype=np.uint8), 2.681517061334488),
    ],
)
def test_crofton_perimeter_matches_scikit_image(mask, expected):
    # Expected values come from skimage.measure.perimeter_crofton(mask, 4).
    assert _crofton_perimeter(mask) == pytest.approx(expected, rel=1e-12)
