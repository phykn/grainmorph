import cv2
import numpy as np


def prepare_mask(mask, target_dim):
    x0, y0, w, h = cv2.boundingRect(mask)
    mask = mask[y0 : y0 + h, x0 : x0 + w]

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x1, y1, w, h = stats[largest, :4]
    mask = (labels[y1 : y1 + h, x1 : x1 + w] == largest).astype(np.uint8)

    # Background 4-connected to the padded border stays empty; the rest are holes.
    flooded = np.pad(mask, 1)
    cv2.floodFill(flooded, None, (0, 0), 2, flags=4)
    mask = (flooded[1:-1, 1:-1] != 2).astype(np.uint8)

    scale = 1.0
    ratio = target_dim / max(w, h)
    width, height = round(w * ratio), round(h * ratio)
    if ratio < 1 and min(width, height) > 1:
        shift = (ratio - 1.0) / 2.0
        mask = cv2.warpAffine(
            mask,
            np.array([[ratio, 0.0, shift], [0.0, ratio, shift]]),
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        scale = 1.0 / ratio

    # OpenCV maps pixel centers, so the half-pixel term is part of the inverse map.
    offset_x = x0 + x1 + (scale - 1.0) / 2.0
    offset_y = y0 + y1 + (scale - 1.0) / 2.0
    return mask, scale, offset_x, offset_y


def restore_coordinates(results, scale, offset_x, offset_y):
    sphericity = results.get("sphericity")
    if sphericity:
        for key in ("inscribed", "enclosing"):
            circle = sphericity[key]
            circle["x"] = float(circle["x"] * scale + offset_x)
            circle["y"] = float(circle["y"] * scale + offset_y)
            circle["r"] = float(circle["r"] * scale)

    aspect_ratio = results.get("aspect_ratio")
    if aspect_ratio:
        ellipse = aspect_ratio["ellipse"]
        ellipse["x"] = float(ellipse["x"] * scale + offset_x)
        ellipse["y"] = float(ellipse["y"] * scale + offset_y)
        for key in ("major", "minor", "w", "h"):
            ellipse[key] = float(ellipse[key] * scale)
        ellipse["bbox"] = [
            [float(x * scale + offset_x), float(y * scale + offset_y)]
            for x, y in ellipse["bbox"]
        ]
