<h1 align="center">GrainMorph</h1>

<p align="center">Shape metrics for a single 2D binary particle mask.</p>

<p align="center">
  <a href="https://pypi.org/project/grainmorph/"><img src="https://img.shields.io/pypi/v/grainmorph" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

- **Wadell roundness**: mean corner radius divided by the inscribed circle radius [1, 2]
- **Circularity**: `4 * pi * area / perimeter^2`
- **Sphericity**: inscribed circle radius divided by enclosing circle radius
- **Aspect ratio**: fitted ellipse major axis divided by minor axis
- Synthetic test masks and a plot of the fitted circles and ellipse

## Install

```bash
pip install grainmorph
```

## Analyze a mask

```python
import grainmorph as gm

mask = gm.utils.create_circle_mask((256, 256), (128, 128), 60)
results = gm.analyze_mask(mask)

print(results["roundness"]["val"])
print(results["circularity"]["val"])
print(results["sphericity"]["val"])
print(results["aspect_ratio"]["val"])
```

`mask` is a 2D `numpy.ndarray` of `bool` or numeric `{0, 1}` values. The largest
4-connected component is measured with its holes filled, and an empty mask
returns `None`.

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

Roundness, circularity, and sphericity lie in `0..1`, and aspect ratio is `>= 1`.
A metric is `None` when the geometry cannot be measured. Coordinates are in
pixels of the input mask.

## Options

```python
results = gm.analyze_mask(
    mask,
    use_aspect_ratio=False,
    roundness_params={"max_dev_thresh": 0.5, "circle_fit_thresh": 0.9},
    eps=0.001,
    target_dim=384,
)
```

- `use_aspect_ratio`, `use_roundness`, `use_circularity`, `use_sphericity`: a metric set to `False` is left out of the result.
- `roundness_params`: overrides `max_dev_thresh` (0.3, boundary discretization), `circle_fit_thresh` (0.98, corner circle fitting), `alpha_ratio` (0.05) and `beta_ratio` (0.001, boundary smoothing).
- `eps`: a metric whose denominator is below this is `None`.
- `target_dim`: a particle larger than this is downscaled for speed, and coordinates are restored to the input scale.

## Plot the result

```python
gm.utils.plot_analysis_results(mask, results)
```

## Create test masks

```python
shape = (256, 256)
circle = gm.utils.create_circle_mask(shape, (128, 128), 60)
ellipse = gm.utils.create_ellipse_mask(shape, (128, 128), 90, 50)
rectangle = gm.utils.create_rectangle_mask(shape, (60, 40), (200, 220))
square = gm.utils.create_square_mask(shape, (60, 60), 120)
pentagon = gm.utils.create_pentagon_mask(shape, (128, 128), 90)
star = gm.utils.create_star_mask(shape, (128, 128), 100, 40, num_points=5)
triangle = gm.utils.create_triangle_mask(shape, (30, 200), (220, 200), (128, 40))
```

Shapes are `(height, width)`, and centers and rectangle corners are `(y, x)`.
Triangle vertices are `(x, y)`.

```python
mask, info = gm.utils.create_particle_mask(
    (256, 256), (128, 128), radius=60, sphericity=0.85, roundness=0.5,
    seed=42, return_info=True,
)
print(info["roundness_achieved"], info["target_met"])
```

`create_particle_mask` roughens an ellipse until its measured roundness
approaches the target. `info["roundness_achieved"]` is the roundness
`analyze_mask` gives for the returned mask.

## Develop

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python -m ruff check --isolated .
```

```bash
python -m ruff format --isolated --check src tests
```

Pushing a `vX.Y.Z` tag publishes to PyPI. The tag must match the version in
`pyproject.toml`.

## License

MIT

## References

1. H. Wadell, "Volume, shape, and roundness of rock particles," The Journal of Geology, 40(5), 443-451, 1932.
2. J. Zheng and R. D. Hryciw, "Traditional soil particle sphericity, roundness and surface roughness by computational geometry," Géotechnique, 65(6), 494-506, 2015.
