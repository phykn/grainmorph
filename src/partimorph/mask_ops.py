from dataclasses import dataclass

import cv2
import numpy as np

from .schema import Mask


@dataclass(frozen=True)
class PreparedMask:
    mask: Mask
    offset_x: float
    offset_y: float
    scale: float


def prepare_mask(mask: Mask, target_dim: int) -> PreparedMask:
    from scipy import ndimage

    cropped, crop_x, crop_y = crop_mask(mask, pad=0)
    mask_bool = cropped.astype(bool, copy=False)

    labeled_mask, num_features = ndimage.label(mask_bool)
    if num_features > 1:
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0
        mask_bool = labeled_mask == np.argmax(sizes)

    filled_mask = ndimage.binary_fill_holes(mask_bool).astype(np.uint8)
    component, component_x, component_y = crop_mask(filled_mask, pad=0)

    source_height, source_width = component.shape
    scale = 1.0
    if max(source_height, source_width) > target_dim:
        resize_scale = target_dim / max(source_height, source_width)
        target_width = max(1, int(round(source_width * resize_scale)))
        target_height = max(1, int(round(source_height * resize_scale)))
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
