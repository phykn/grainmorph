# PartiMorph

PartiMorph analyzes a single 2D binary particle mask and returns geometric shape metrics.

| Metric | Meaning |
|---|---|
| Wadell roundness | Corner curvature-based roundness |
| Circularity | `4 * pi * area / perimeter^2` |
| Sphericity | Inscribed circle radius divided by enclosing circle radius |
| Aspect ratio | Fitted ellipse major axis divided by minor axis |

Roundness, circularity, and sphericity are clipped to `0..1`. Aspect ratio is `>= 1`.

## Installation

PartiMorph requires Python `3.12+`.

```bash
pip install partimorph
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import partimorph as pm

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
- The largest connected foreground component is selected and holes are filled
- Surrounding zero padding does not change the measured shape
- Downscaling is skipped if its output would have a one-pixel axis
- Shape and center arguments use image order: `(height, width)` and `(y, x)`

## Project Layout

- `analyzer.py`: public analysis workflow and result-coordinate restoration
- `validation.py`: public input normalization and analysis-option validation
- `mask_ops.py`: component selection, crop/resize transforms, and contours
- `metrics.py`: metric formulas and public result payloads
- `fitting.py`: circle and ellipse fitting plus oriented bounding boxes
- `wadell/`: boundary, smoothing, discretization, and corner roundness stages
- `utils/`: synthetic mask construction and visualization

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

Visualization:

- `pm.utils.plot_analysis_results(mask, results)`

## Changes in 0.1.8

- Corrected the fitted ellipse overlay orientation for rotated particles.
- Unified inscribed-circle fitting used by sphericity and Wadell roundness.
- Simplified ellipse fitting and bounding-box coordinate transforms while preserving the public API and analysis results.

## License

MIT
