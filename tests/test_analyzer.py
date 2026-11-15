import numpy as np
import pytest

import grainmorph as gm
from grainmorph.metrics import compute_roundness


@pytest.mark.parametrize(
    "enabled",
    [
        (),
        ("aspect_ratio",),
        ("roundness",),
        ("circularity",),
        ("sphericity",),
        ("aspect_ratio", "roundness", "circularity", "sphericity"),
    ],
)
def test_metric_selection(enabled):
    mask = gm.utils.create_circle_mask((100, 100), (50, 50), 40)
    keys = ("aspect_ratio", "roundness", "circularity", "sphericity")

    results = gm.analyze_mask(mask, **{f"use_{key}": key in enabled for key in keys})

    assert set(results) == set(enabled)
    for key, result in results.items():
        if key == "aspect_ratio":
            assert result["val"] == pytest.approx(1.0, abs=0.01)
        else:
            assert 0.9 < result["val"] <= 1.0


def test_empty_mask_returns_none():
    assert gm.analyze_mask(np.zeros((8, 8), dtype=bool)) is None


@pytest.mark.parametrize(
    ("mask", "error", "match"),
    [
        ([[0, 1]], TypeError, "numpy.ndarray"),
        (np.zeros((4, 4, 4)), ValueError, "mask must be 2D"),
        (np.array([[0, 2], [1, 0]]), ValueError, "binary with values in {0, 1}"),
        (np.array([[0.0, np.nan]]), ValueError, "binary with values in {0, 1}"),
        (np.array([["a", "b"]]), ValueError, "binary with values in {0, 1}"),
    ],
)
def test_rejects_invalid_mask(mask, error, match):
    with pytest.raises(error, match=match):
        gm.analyze_mask(mask)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"eps": 0.0}, ValueError, "eps must be > 0"),
        ({"eps": np.nan}, ValueError, "eps must be > 0"),
        ({"eps": np.inf}, ValueError, "eps must be > 0"),
        ({"target_dim": 0}, ValueError, "target_dim must be > 0"),
        ({"target_dim": 1.5}, TypeError, "target_dim must be an integer"),
        ({"target_dim": True}, TypeError, "target_dim must be an integer"),
    ],
)
def test_rejects_invalid_options(options, error, match):
    with pytest.raises(error, match=match):
        gm.analyze_mask(np.ones((8, 8), dtype=bool), **options)


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int32, np.float64])
def test_accepts_binary_dtypes(dtype):
    mask = np.zeros((20, 30), dtype=dtype)
    mask[5:14, 10:19] = 1

    results = gm.analyze_mask(mask, use_roundness=False)

    assert results["sphericity"]["inscribed"] == {"x": 14.0, "y": 9.0, "r": 5.0}


def test_roundness_params_reach_the_metric():
    mask = gm.utils.create_star_mask((200, 200), (100, 100), 80, 40).astype(np.uint8)
    params = {"max_dev_thresh": 0.5, "alpha_ratio": 0.025, "beta_ratio": 0.0005}
    rows, cols = np.nonzero(mask)
    cropped = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    results = gm.analyze_mask(mask, roundness_params=params)

    assert results["roundness"] == compute_roundness(cropped, **params)
    assert results["roundness"] != gm.analyze_mask(mask)["roundness"]


def test_resize_restores_coordinates():
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[100:900, 200:800] = True

    results = gm.analyze_mask(mask, use_roundness=False)

    inscribed = results["sphericity"]["inscribed"]
    assert inscribed["x"] == pytest.approx(500, abs=5)
    assert inscribed["r"] == pytest.approx(300, abs=3)
    enclosing = results["sphericity"]["enclosing"]
    np.testing.assert_allclose([enclosing["x"], enclosing["y"]], [499.5, 499.5], atol=2)
    assert enclosing["r"] == pytest.approx(500, abs=3)
    ellipse = results["aspect_ratio"]["ellipse"]
    np.testing.assert_allclose(np.min(ellipse["bbox"], axis=0), [200, 100], atol=2)
    np.testing.assert_allclose(np.max(ellipse["bbox"], axis=0), [799, 899], atol=2)
    assert ellipse["major"] == max(ellipse["w"], ellipse["h"])
    assert ellipse["major"] == pytest.approx(799, rel=0.01)


@pytest.mark.parametrize("shape", [(1, 1000), (1000, 1)])
def test_one_pixel_wide_masks_are_not_resized(shape):
    results = gm.analyze_mask(np.ones(shape, dtype=bool), use_roundness=False)

    assert results["aspect_ratio"] is None
    assert results["circularity"]["val"] < 0.01
    assert results["sphericity"]["enclosing"]["r"] == pytest.approx(499.5, abs=0.01)


def test_resize_keeps_thin_axis():
    mask = np.zeros((1200, 1200), dtype=bool)
    mask[100:1100, 550:553] = True
    options = {
        "use_roundness": False,
        "use_circularity": False,
        "use_sphericity": False,
    }

    resized = gm.analyze_mask(mask, **options)
    reference = gm.analyze_mask(mask, target_dim=2000, **options)

    np.testing.assert_allclose(
        resized["aspect_ratio"]["val"], reference["aspect_ratio"]["val"]
    )


@pytest.mark.parametrize("target_dim", [32, 384])
def test_results_follow_translation(target_dim):
    mask = gm.utils.create_ellipse_mask((100, 120), (50, 60), 35, 20)
    shifted = np.pad(mask, ((23, 500), (37, 0)))

    reference = gm.analyze_mask(mask, target_dim=target_dim)
    actual = gm.analyze_mask(shifted, target_dim=target_dim)

    for key in reference:
        np.testing.assert_allclose(actual[key]["val"], reference[key]["val"])
    for key in ("inscribed", "enclosing"):
        circle = actual["sphericity"][key]
        original = reference["sphericity"][key]
        np.testing.assert_allclose(
            [circle["x"], circle["y"], circle["r"]],
            [original["x"] + 37, original["y"] + 23, original["r"]],
        )
    ellipse = actual["aspect_ratio"]["ellipse"]
    original = reference["aspect_ratio"]["ellipse"]
    np.testing.assert_allclose(
        [ellipse["x"], ellipse["y"]], [original["x"] + 37, original["y"] + 23]
    )
    for key in ("major", "minor", "w", "h", "angle"):
        np.testing.assert_allclose(ellipse[key], original[key])
    np.testing.assert_allclose(ellipse["bbox"], np.add(original["bbox"], [37, 23]))
