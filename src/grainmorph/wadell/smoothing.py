import cv2
import numpy as np


def smooth_boundary(points, alpha_ratio, beta_ratio):
    if not (np.isfinite(alpha_ratio) and np.isfinite(beta_ratio)):
        raise ValueError("smoothing ratios must be finite.")

    num = len(points)
    perimeter = cv2.arcLength(points.astype(np.float32).reshape(-1, 1, 2), closed=True)
    coefs = alpha_ratio * perimeter * np.array([-2, 1, 0, 0])
    coefs = coefs + beta_ratio * perimeter * np.array([-6, 4, -1, 0])
    idx = np.arange(num)
    kernel = -coefs[np.minimum(np.minimum(idx, num - idx), 3)]
    kernel[0] += 1.0

    # The system is circulant, so it is solved in Fourier space; modes with a
    # near-zero eigenvalue are dropped, as a least-squares solve would drop them.
    eigenvalues = np.fft.fft(kernel)
    transformed = np.fft.fft(points, axis=0)
    usable = np.abs(eigenvalues) > np.finfo(float).eps * num * np.abs(eigenvalues).max()
    solution = np.zeros_like(transformed)
    np.divide(transformed, eigenvalues[:, None], out=solution, where=usable[:, None])
    return np.fft.ifft(solution, axis=0).real
