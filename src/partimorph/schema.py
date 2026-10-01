from typing import Annotated, TypeAlias, TypedDict

import numpy as np

Bbox: TypeAlias = list[list[float]]
Mask: TypeAlias = Annotated[np.ndarray, "shape=(H, W), dtype=uint8"]
Coordinates: TypeAlias = Annotated[np.ndarray, "shape=(N, 2), dtype=float64"]
Boundary: TypeAlias = Annotated[np.ndarray, "shape=(N, 2), dtype=int32"]
Points: TypeAlias = Annotated[np.ndarray, "shape=(N, 2), dtype=float32"]


class CircleData(TypedDict):
    x: float
    y: float
    r: float


class EllipseData(TypedDict):
    major: float
    minor: float
    x: float
    y: float
    angle: float
    w: float
    h: float
    bbox: Bbox


class RoundnessResult(TypedDict):
    val: float


class CircularityResult(TypedDict):
    val: float


class SphericityResult(TypedDict):
    val: float
    inscribed: CircleData
    enclosing: CircleData


class AspectRatioResult(TypedDict):
    val: float
    ellipse: EllipseData


class AnalysisResult(TypedDict, total=False):
    roundness: RoundnessResult | None
    circularity: CircularityResult | None
    sphericity: SphericityResult | None
    aspect_ratio: AspectRatioResult | None
