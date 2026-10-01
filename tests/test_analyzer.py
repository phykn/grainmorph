import numpy as np
import pytest

import partimorph as pm


def test_analyze_mask_full():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    results = pm.analyze_mask(mask)

    assert results is not None
    assert set(results) == {
        "roundness",
        "circularity",
        "sphericity",
        "aspect_ratio",
    }
    assert results["aspect_ratio"] is not None
    assert np.isclose(results["aspect_ratio"]["val"], 1.0, atol=0.1)


def test_metrics_consistency():
    y, x = np.ogrid[:100, :100]
    mask = (x - 50) ** 2 + (y - 50) ** 2 <= 40**2

    results = pm.analyze_mask(mask)

    assert results is not None
    assert results["circularity"]["val"] > 0.9
    assert results["sphericity"]["val"] > 0.9
    assert results["roundness"]["val"] > 0.9


@pytest.mark.parametrize("eps", [0.0, -1.0, np.nan, np.inf])
def test_analyze_mask_rejects_invalid_eps(eps):
    mask = np.ones((8, 8), dtype=bool)

    with pytest.raises(ValueError, match="eps must be > 0"):
        pm.analyze_mask(mask, eps=eps)


@pytest.mark.parametrize("target_dim", [0, -1])
def test_analyze_mask_rejects_non_positive_target_dim(target_dim):
    mask = np.ones((8, 8), dtype=bool)

    with pytest.raises(ValueError, match="target_dim must be > 0"):
        pm.analyze_mask(mask, target_dim=target_dim)


@pytest.mark.parametrize("target_dim", [1.5, True])
def test_analyze_mask_rejects_non_integer_target_dim(target_dim):
    mask = np.ones((8, 8), dtype=bool)

    with pytest.raises(TypeError, match="target_dim must be an integer"):
        pm.analyze_mask(mask, target_dim=target_dim)


def test_analyze_mask_uses_largest_component():
    mask = np.zeros((120, 120), dtype=bool)
    mask[10:30, 10:30] = True
    mask[50:110, 50:110] = True

    results = pm.analyze_mask(mask, use_roundness=False, use_circularity=False)

    assert results is not None
    assert results["sphericity"] is not None
    center = results["sphericity"]["inscribed"]
    assert abs(center["x"] - 80) < 5
    assert abs(center["y"] - 80) < 5


def test_analyze_mask_resize_path_scales_coordinates():
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[100:900, 200:800] = True

    results = pm.analyze_mask(mask, target_dim=384, use_roundness=False)

    assert results is not None
    assert results["sphericity"] is not None
    assert results["aspect_ratio"] is not None

    inscribed = results["sphericity"]["inscribed"]
    assert abs(inscribed["x"] - 500) < 5
    assert abs(inscribed["y"] - 500) < 110

    ellipse = results["aspect_ratio"]["ellipse"]
    assert np.isclose(ellipse["major"], max(ellipse["w"], ellipse["h"]))
    assert np.isclose(ellipse["minor"], min(ellipse["w"], ellipse["h"]))
    assert len(ellipse["bbox"]) == 4
    assert all(len(point) == 2 for point in ellipse["bbox"])


@pytest.mark.parametrize("shape", [(1, 1000), (1000, 1)])
def test_analyze_mask_resizes_thin_masks_without_crashing(shape):
    results = pm.analyze_mask(
        np.ones(shape, dtype=bool),
        use_roundness=False,
    )

    assert results is not None
    assert results["circularity"] is not None
    assert results["sphericity"] is not None


def test_analyze_mask_does_not_collapse_non_degenerate_axis():
    mask = np.zeros((1200, 1200), dtype=bool)
    mask[100:1100, 550:553] = True

    resized = pm.analyze_mask(
        mask,
        use_roundness=False,
        use_circularity=False,
        use_sphericity=False,
    )
    reference = pm.analyze_mask(
        mask,
        target_dim=2000,
        use_roundness=False,
        use_circularity=False,
        use_sphericity=False,
    )

    assert resized is not None
    assert reference is not None
    assert resized["aspect_ratio"] is not None
    assert reference["aspect_ratio"] is not None
    np.testing.assert_allclose(
        resized["aspect_ratio"]["val"],
        reference["aspect_ratio"]["val"],
    )


def test_metrics_are_invariant_to_zero_padding():
    small = pm.utils.create_circle_mask((100, 100), (50, 50), 2)
    large = pm.utils.create_circle_mask((2000, 2000), (1000, 1000), 2)
    options = {"use_roundness": False, "use_aspect_ratio": False}

    small_results = pm.analyze_mask(small, **options)
    large_results = pm.analyze_mask(large, **options)

    assert small_results is not None
    assert large_results is not None
    np.testing.assert_allclose(
        small_results["circularity"]["val"],
        large_results["circularity"]["val"],
    )
    np.testing.assert_allclose(
        small_results["sphericity"]["val"],
        large_results["sphericity"]["val"],
    )
    for circle in ("inscribed", "enclosing"):
        small_circle = small_results["sphericity"][circle]
        large_circle = large_results["sphericity"][circle]
        np.testing.assert_allclose(small_circle["r"], large_circle["r"])
        np.testing.assert_allclose(large_circle["x"] - small_circle["x"], 950)
        np.testing.assert_allclose(large_circle["y"] - small_circle["y"], 950)


@pytest.mark.parametrize("target_dim", [32, 384])
def test_analysis_geometry_is_invariant_to_translation(target_dim):
    mask = pm.utils.create_ellipse_mask((100, 120), (50, 60), 35, 20)
    shifted = np.pad(mask, ((23, 0), (37, 0)))

    reference = pm.analyze_mask(mask, target_dim=target_dim)
    actual = pm.analyze_mask(shifted, target_dim=target_dim)

    for key in reference:
        assert actual[key] is not None
        np.testing.assert_allclose(actual[key]["val"], reference[key]["val"])

    for key in ("inscribed", "enclosing"):
        circle = actual["sphericity"][key]
        original = reference["sphericity"][key]
        np.testing.assert_allclose(
            [circle["x"] - original["x"], circle["y"] - original["y"]], [37, 23]
        )
        np.testing.assert_allclose(circle["r"], original["r"])

    ellipse = actual["aspect_ratio"]["ellipse"]
    original = reference["aspect_ratio"]["ellipse"]
    np.testing.assert_allclose(
        [ellipse["x"] - original["x"], ellipse["y"] - original["y"]], [37, 23]
    )
    for key in ("major", "minor", "w", "h", "angle"):
        np.testing.assert_allclose(ellipse[key], original[key])
    np.testing.assert_allclose(ellipse["bbox"], np.asarray(original["bbox"]) + [37, 23])


@pytest.mark.parametrize(
    "enabled",
    [(), ("aspect_ratio",), ("roundness",), ("circularity",), ("sphericity",)],
)
def test_analysis_metric_selection(enabled):
    mask = pm.utils.create_circle_mask((80, 80), (40, 40), 20)
    keys = ("aspect_ratio", "roundness", "circularity", "sphericity")
    options = {f"use_{key}": key in enabled for key in keys}

    results = pm.analyze_mask(mask, **options)

    assert set(results) == set(enabled)
    assert all(result is not None for result in results.values())
