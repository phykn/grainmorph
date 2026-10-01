import json

import numpy as np
import pytest

import partimorph as pm


def test_create_particle_mask():
    mask = pm.utils.create_particle_mask(
        shape=(256, 256),
        center=(128, 128),
        radius=50,
        sphericity=0.8,
        roundness=0.7,
    )

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (256, 256)
    assert mask.dtype == bool
    assert np.any(mask)
    y_coords, x_coords = np.where(mask)
    assert np.abs(np.mean(y_coords) - 128) < 5
    assert np.abs(np.mean(x_coords) - 128) < 5


def test_create_particle_mask_zero_radius():
    mask, info = pm.utils.create_particle_mask(
        shape=(100, 100),
        center=(50, 50),
        radius=0.0,
        sphericity=0.8,
        roundness=0.8,
        return_info=True,
    )

    assert np.count_nonzero(mask) <= 1
    assert info["roundness_achieved"] == 0.0


def test_roundness_rough_particle_golden():
    mask = pm.utils.create_particle_mask(
        shape=(256, 256),
        center=(128, 128),
        radius=60,
        sphericity=0.85,
        roundness=0.5,
        seed=42,
    )

    results = pm.analyze_mask(mask)

    assert results is not None
    assert results["roundness"] is not None
    assert 0.3 < results["roundness"]["val"] < 0.7


def test_particle_info_measures_returned_mask():
    mask, info = pm.utils.create_particle_mask(
        shape=(96, 96),
        center=(0, 0),
        radius=25,
        sphericity=0.8,
        roundness=0.6,
        seed=7,
        return_info=True,
    )
    results = pm.analyze_mask(
        mask,
        use_aspect_ratio=False,
        use_circularity=False,
        use_sphericity=False,
    )

    assert results is not None
    actual = results["roundness"]["val"] if results["roundness"] else 0.0
    np.testing.assert_allclose(info["roundness_achieved"], actual)
    assert isinstance(info["target_met"], bool)
    json.dumps(info)


def test_mask_generators_reject_non_finite_radii():
    with pytest.raises(ValueError, match="finite"):
        pm.utils.create_circle_mask((16, 16), (8, 8), np.nan)
    with pytest.raises(ValueError, match="finite"):
        pm.utils.create_ellipse_mask((16, 16), (8, 8), np.inf, 2)
    with pytest.raises(ValueError, match="finite"):
        pm.utils.create_pentagon_mask((16, 16), (8, 8), np.nan)
    with pytest.raises(ValueError, match="finite"):
        pm.utils.create_star_mask((16, 16), (8, 8), 5, np.inf)


@pytest.mark.parametrize("field", ["sphericity", "roundness"])
def test_create_particle_mask_rejects_non_finite_targets(field):
    options = {"sphericity": 0.8, "roundness": 0.6}
    options[field] = np.nan

    with pytest.raises(ValueError, match="finite"):
        pm.utils.create_particle_mask(
            shape=(32, 32),
            center=(16, 16),
            radius=8,
            **options,
        )


@pytest.mark.parametrize("radius", [0, 20])
def test_particle_report_preserves_seeded_mask(radius):
    options = {
        "shape": (80, 80),
        "center": (40, 40),
        "radius": radius,
        "sphericity": 0.8,
        "roundness": 0.6,
        "seed": 42,
    }

    mask = pm.utils.create_particle_mask(**options)
    reported, info = pm.utils.create_particle_mask(**options, return_info=True)

    np.testing.assert_array_equal(mask, reported)
    assert info["target_met"] == (info["abs_error"] <= info["report_tol"])
    assert info["abs_error"] == abs(
        info["roundness_target"] - info["roundness_achieved"]
    )
    json.dumps(info)
