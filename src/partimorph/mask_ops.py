from dataclasses import dataclass

import cv2
import numpy as np

from .schema import AnalysisResult, Mask


@dataclass(frozen=True)
class PreparedMask:
    mask: Mask
    offset_x: float
    offset_y: float
    scale: float

    def restore_results(self, results: AnalysisResult) -> None:
        sphericity = results.get("sphericity")
        if sphericity:
            for key in ("inscribed", "enclosing"):
                circle = sphericity[key]
                circle["x"] = float(circle["x"] * self.scale + self.offset_x)
                circle["y"] = float(circle["y"] * self.scale + self.offset_y)
                circle["r"] = float(circle["r"] * self.scale)

        aspect_ratio = results.get("aspect_ratio")
        if aspect_ratio:
            ellipse = aspect_ratio["ellipse"]
            ellipse["x"] = float(ellipse["x"] * self.scale + self.offset_x)
            ellipse["y"] = float(ellipse["y"] * self.scale + self.offset_y)
            for key in ("major", "minor", "w", "h"):
                ellipse[key] = float(ellipse[key] * self.scale)
            ellipse["bbox"] = [
                [
                    float(x * self.scale + self.offset_x),
                    float(y * self.scale + self.offset_y),
                ]
                for x, y in ellipse["bbox"]
            ]


def prepare_mask(mask: Mask, target_dim: int) -> PreparedMask:
    cropped, crop_x, crop_y = crop_mask(mask, pad=0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cropped, connectivity=4
    )
    if num_labels > 2:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cropped = (labels == largest).astype(np.uint8)

    # Background 4-connected to the padded border stays empty; the rest are holes.
    flooded = np.pad(cropped, 1)
    cv2.floodFill(flooded, None, (0, 0), 2, flags=4)
    filled_mask = (flooded[1:-1, 1:-1] != 2).astype(np.uint8)
    component, component_x, component_y = crop_mask(filled_mask, pad=0)

    source_height, source_width = component.shape
    scale = 1.0
    if max(source_height, source_width) > target_dim:
        resize_scale = target_dim / max(source_height, source_width)
        target_width = max(1, round(source_width * resize_scale))
        target_height = max(1, round(source_height * resize_scale))
        if target_width > 1 and target_height > 1:
            translation = (resize_scale - 1.0) / 2.0
            transform = np.array(
                [
                    [resize_scale, 0.0, translation],
                    [0.0, resize_scale, translation],
                ],
                dtype=np.float64,
            )
            component = cv2.warpAffine(
                component,
                transform,
                (target_width, target_height),
                flags=cv2.INTER_NEAREST,
            )
            scale = 1.0 / resize_scale

    # OpenCV maps pixel centers, so the half-pixel term is part of the inverse map.
    offset_x = crop_x + component_x + (scale - 1.0) / 2.0
    offset_y = crop_y + component_y + (scale - 1.0) / 2.0

    return PreparedMask(
        mask=component,
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        scale=float(scale),
    )


def get_contours(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    cropped_mask, offset_x, offset_y = crop_mask(mask, pad=1)

    if cropped_mask.size == 0:
        return ()

    if cropped_mask.dtype != np.uint8:
        cropped_mask = cropped_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    offset = np.array([[[offset_x, offset_y]]], dtype=np.int32)
    return tuple(contour + offset for contour in contours)


def crop_mask(mask: np.ndarray, pad: int = 1) -> tuple[np.ndarray, int, int]:
    if mask.dtype != np.uint8:
        binary_mask_uint8 = mask.astype(np.uint8)
    else:
        binary_mask_uint8 = mask

    x_min, y_min, width, height = cv2.boundingRect(binary_mask_uint8)

    if width == 0 or height == 0:
        return (np.zeros((0, 0), dtype=mask.dtype), 0, 0)

    cropped = mask[y_min : y_min + height, x_min : x_min + width]

    if pad > 0:
        cropped = np.pad(cropped, pad_width=pad, mode="constant", constant_values=0)

    offset_y = y_min - pad
    offset_x = x_min - pad

    return (cropped, offset_x, offset_y)
