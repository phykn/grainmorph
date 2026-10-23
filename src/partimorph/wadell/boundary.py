import cv2
import numpy as np


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float64)
