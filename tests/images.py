"""Synthetic test images. Real X-rays are not committed to the repo."""

import cv2
import numpy as np


def xray_like(size=512):
    """Smooth grayscale image shaped roughly like a frontal chest X-ray."""
    img = np.full((size, size), 20, np.float32)
    c = size // 2
    cv2.ellipse(img, (c, c + size // 10), (size * 2 // 5, size * 9 // 20), 0, 0, 360, 170, -1)
    for dx in (-size // 6, size // 6):
        cv2.ellipse(img, (c + dx, c), (size // 9, size // 4), 0, 0, 360, 70, -1)
    cv2.rectangle(img, (c - size // 30, 0), (c + size // 30, size), 200, -1)
    img = cv2.GaussianBlur(img, (0, 0), size / 60)
    img += np.random.default_rng(0).normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def tinted(gray, bgr_scale=(1.0, 0.85, 0.6)):
    """Grayscale image tinted one colour, like a photo of film on a lightbox."""
    return np.clip(gray[..., None] * np.array(bgr_scale), 0, 255).astype(np.uint8)


def noise(size=512, channels=1):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (size, size, channels), dtype=np.uint8)
    return img[..., 0] if channels == 1 else img


def colourful(size=512):
    """Smooth image with several distinct hues, like a colour photo."""
    img = np.zeros((size, size, 3), np.uint8)
    h = size // 2
    img[:h, :h] = (40, 40, 220)
    img[:h, h:] = (40, 200, 40)
    img[h:, :h] = (220, 60, 40)
    img[h:, h:] = (40, 220, 230)
    return cv2.GaussianBlur(img, (0, 0), size / 100)


def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
