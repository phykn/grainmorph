# GrainMorph

GrainMorph analyzes a single 2D binary particle mask and returns geometric shape metrics.

| Metric | Meaning |
|---|---|
| Wadell roundness | Corner curvature-based roundness |
| Circularity | `4 * pi * area / perimeter^2` |
| Sphericity | Inscribed circle radius divided by enclosing circle radius |
| Aspect ratio | Fitted ellipse major axis divided by minor axis |

Roundness, circularity, and sphericity are clipped to `0..1`. Aspect ratio is `>= 1`.

## Installation

GrainMorph requires Python `3.12+`.

```bash
pip install grainmorph
```

For local development:

```bash
pip install -e ".[dev]"
```

Run the tests and code checks with the same Python environment:

```bash
python -m pytest -q
python -m ruff check --isolated .
python -m ruff format --isolated --check src tests
```

## Releasing

Pushing a `vX.Y.Z` tag publishes that version to PyPI through
`.github/workflows/publish.yml`, using PyPI trusted publishing (no API token):

1. Set the same version in `pyproject.toml` and `src/grainmorph/__init__.py`.
2. Commit, then tag and push: `git tag -a vX.Y.Z -m "Release X.Y.Z" && git push origin main vX.Y.Z`.

The workflow fails if the tag and the `pyproject.toml` version differ, then runs the tests before uploading.
PyPI must list this repository as a trusted publisher for `grainmorph`
(owner `phykn`, repository `grainmorph`, workflow `publish.yml`, environment `pypi`).

## Quick Start

```python
import grainmorph as pm

mask = pm.utils.create_circle_mask((256, 256), (128, 128), 60)
results = pm.analyze_mask(mask)

if results is None:
    raise ValueError("mask has no foreground pixels")

print(results["roundness"]["val"])
print(results["circularity"]["val"])
print(results["sphericity"]["val"])
print(results["aspect_ratio"]["val"])
```

Visualize the results:

```python
pm.utils.plot_analysis_results(mask, results)
```

## API

```python
pm.analyze_mask(
    mask,
    use_aspect_ratio=True,
    use_roundness=True,
    use_circularity=True,
    use_sphericity=True,
    roundness_params=None,
    eps=0.001,
    target_dim=384,
)
```

| Parameter | Description |
|---|---|
| `mask` | 2D `numpy.ndarray` with `bool` or `{0, 1}` values |
| `use_*` | Toggle individual metrics on/off |
| `roundness_params` | Optional dict to override Wadell roundness parameters |
| `eps` | Tolerance for geometric computations |
| `target_dim` | Particle bounds larger than this are downscaled for speed, then coordinates are restored |

`roundness_params` accepts these Wadell options. Defaults are defined in
`wadell/roundness.py` and also apply to synthetic particle calibration:

| Option | Default | Stage |
|---|---|---|
| `max_dev_thresh` | `0.3` | Boundary discretization |
| `circle_fit_thresh` | `0.98` | Corner circle fitting |
| `alpha_ratio` | `0.05` | Boundary smoothing |
| `beta_ratio` | `0.001` | Boundary smoothing |

Returns `None` for an empty mask. Otherwise, returns an `AnalysisResult` dict. Disabled metrics are omitted; enabled metrics are present and may be `None` when the geometry cannot be measured.

## Result Shape

```python
{
  "roundness": {"val": float} | None,
  "circularity": {"val": float} | None,
  "sphericity": {
    "val": float,
    "inscribed": {"x": float, "y": float, "r": float},
    "enclosing": {"x": float, "y": float, "r": float},
  } | None,
  "aspect_ratio": {
    "val": float,
    "ellipse": {
      "major": float, "minor": float,
      "x": float, "y": float, "angle": float,
      "w": float, "h": float, "bbox": list[list[float]],
    },
  } | None,
}
```

## Input Rules

- `mask` must be a 2D `numpy.ndarray`
- Allowed values are `bool` or numeric `{0, 1}`
- Floating masks must not contain `NaN` or `inf`
- Empty mask returns `None`
- The largest 4-connected foreground component is selected and holes are filled
- Surrounding zero padding does not change the measured shape
- Downscaling is skipped if its output would have a one-pixel axis
- Shape, center, and rectangle corner arguments use image order: `(height, width)` and `(y, x)`
- Triangle vertices use Cartesian order `(x, y)`, matching polygon coordinates

## Project Layout

- `analyzer.py`: public analysis workflow and metric selection
- `validation.py`: public input normalization and analysis-option validation
- `mask_ops.py`: component selection, contours, and `PreparedMask` crop/resize transforms with result-coordinate restoration
- `metrics.py`: metric formulas and public result payloads
- `fitting.py`: circle and ellipse fitting plus oriented bounding boxes
- `wadell/roundness.py`: roundness options and the boundary-to-radius workflow
- `wadell/boundary.py` and `smoothing.py`: closed boundary extraction and periodic smoothing
- `wadell/discretize.py`: boundary approximation with `(x, y)` point arrays
- `wadell/corner.py`: convex corner selection, circle fitting, and accepted corner radii
- `utils/`: basic mask construction, parametric particle calibration and reporting, and visualization

## Utilities

Mask constructors:

- `pm.utils.create_circle_mask(...)`
- `pm.utils.create_ellipse_mask(...)`
- `pm.utils.create_rectangle_mask(...)`
- `pm.utils.create_square_mask(...)`
- `pm.utils.create_triangle_mask(...)`
- `pm.utils.create_pentagon_mask(...)`
- `pm.utils.create_star_mask(...)`
- `pm.utils.create_particle_mask(...)`

For example, a triangle in a non-square image uses `(x, y)` vertices:

```python
mask = pm.utils.create_triangle_mask((20, 60), (30, 2), (40, 2), (30, 8))
```

`create_particle_mask` attempts to reach the requested roundness by calibrating
surface roughness. Set `seed` for reproducible masks and `return_info=True` to
receive `(mask, info)`. The report measures the returned mask, including any
clipping at the image bounds; `target_met` compares its roundness error with
`report_tol`. A zero-radius particle reports roundness `0.0`.

Visualization:

- `pm.utils.plot_analysis_results(mask, results)`

## License

MIT
