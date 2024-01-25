import numpy as np


def smooth_labels(y, sigma=2):
    # Ensure y is a 1D array
    y = y.flatten()
    smoothed_y = np.zeros_like(y, dtype=float)
    for idx in np.where(y == 1)[0]:
        gaussian = np.exp(-0.5 * ((np.arange(len(y)) - idx) / sigma) ** 2)
        gaussian /= gaussian.sum()
        smoothed_y += 5.1*gaussian  # Both are 1D arrays, so this should work
    smoothed_y = np.clip(smoothed_y, None, 1)
    return smoothed_y