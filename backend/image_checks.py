"""Cheap checks that reject obvious non-X-ray uploads before the model sees them.

The model gives a confident answer for any input, including random noise, so
these filter out the clearest junk. They do not detect other kinds of X-ray
or grayscale photos. Thresholds were set on public chest X-rays (including
lightbox photos with a blue or sepia tint) and on colour photos and noise.
"""

import cv2
import numpy as np

SIZE = 224
MIN_STD = 10.0  # real X-rays measured >= 31
MAX_ROUGHNESS = 0.3  # real X-rays <= 0.18, noise >= 0.8
MAX_COLOUR_RESIDUAL = 14.0  # tinted X-rays <= 12.5, bright photos up to 35


def rejection_reason(image):
    """Return "flat", "noisy" or "colour" if the image is rejected, else None.

    `image` is a uint8 array as decoded by OpenCV: 2D grayscale or 3-channel BGR.
    """
    small = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = small.mean(axis=2) if small.ndim == 3 else small

    std = gray.std()
    if std < MIN_STD:
        return "flat"

    # Average jump between neighbouring pixels, relative to overall contrast.
    # X-rays change smoothly; noise jumps by about as much as its contrast.
    step = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
    if step / std > MAX_ROUGHNESS:
        return "noisy"

    if small.ndim == 3 and colour_residual(small, gray) > MAX_COLOUR_RESIDUAL:
        return "colour"

    return None


def colour_residual(bgr, gray):
    """Colour left over after modelling each channel as a linear function of brightness.

    A tinted grayscale image (one hue at different brightnesses) scores near
    zero; an image with several distinct hues scores high.
    """
    x = gray.ravel()
    design = np.stack([x, np.ones_like(x)], axis=1)
    residuals = []
    for c in range(3):
        y = bgr[:, :, c].ravel()
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals.append(np.sqrt(np.mean((y - design @ coef) ** 2)))
    return float(np.mean(residuals))
