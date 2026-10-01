import numpy as np
import pytest

import partimorph as pm
from partimorph.wadell import compute_roundness


@pytest.mark.parametrize(
    "params",
    [{}, {"max_dev_thresh": 0.5, "alpha_ratio": 0.025, "beta_ratio": 0.0005}],
)
def test_analysis_roundness_uses_wadell_options(params):
    mask = pm.utils.create_circle_mask((100, 100), (50, 50), 30)
    expected = compute_roundness(mask, **params)

    results = pm.analyze_mask(
        mask,
        use_aspect_ratio=False,
        use_circularity=False,
        use_sphericity=False,
        roundness_params=params,
    )

    assert expected is not None
    np.testing.assert_allclose(results["roundness"]["val"], np.clip(expected, 0.0, 1.0))
