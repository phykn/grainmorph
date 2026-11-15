import json

import numpy as np
import pytest

import grainmorph as gm


def _measure_roundness(mask):
    results = gm.analyze_mask(mask)
    return results["roundness"]["val"]


def test_particle_reaches_target_roundness():
    options = {"shape": (256, 256), "center": (128, 128), "radius": 60, "seed": 42}

    mask, info = gm.utils.create_particle_mask(
        sphericity=0.85, roundness=0.5, return_info=True, **options
    )

    assert mask.shape == (256, 256) and mask.dtype == bool
    y, x = np.nonzero(mask)
    np.testing.assert_allclose([y.mean(), x.mean()], [128, 128], atol=5)
    assert info["roundness_target"] == 0.5
    assert info["abs_error"] == abs(0.5 - info["roundness_achieved"]) < 0.1
    assert info["target_met"] is True
    assert info["amplitude"] > 0
    json.dumps(info)
    np.testing.assert_array_equal(
        mask, gm.utils.create_particle_mask(sphericity=0.85, roundness=0.5, **options)
    )
    smooth = gm.utils.create_particle_mask(sphericity=0.85, roundness=1.0, **options)
    assert _measure_roundness(smooth) > _measure_roundness(mask) + 0.2


@pytest.mark.parametrize(
    ("shape", "center", "radius"),
    [((96, 96), (0, 0), 25), ((800, 800), (400, 400), 300)],
)
def test_particle_info_measures_returned_mask(shape, center, radius):
    mask, info = gm.utils.create_particle_mask(
        shape, center, radius, 0.8, 0.4, seed=1, return_info=True
    )

    assert info["roundness_achieved"] == _measure_roundness(mask)


def test_zero_radius_particle_is_one_pixel():
    mask, info = gm.utils.create_particle_mask(
        (100, 100), (50, 50), 0, 0.8, 0.8, return_info=True
    )

    assert np.argwhere(mask).tolist() == [[50, 50]]
    assert info["roundness_achieved"] == 0.0
    assert info["amplitude"] == 0.0


@pytest.mark.parametrize(
    "options",
    [
        {"radius": np.inf},
        {"radius": -1},
        {"sphericity": np.nan},
        {"roundness": np.nan},
        {"num_angles": 2},
    ],
)
def test_particle_rejects_invalid_options(options):
    options = {"radius": 8, "sphericity": 0.8, "roundness": 0.6, **options}

    with pytest.raises(ValueError):
        gm.utils.create_particle_mask((32, 32), (16, 16), **options)
