import numpy as np

from image_checks import rejection_reason
from images import colourful, noise, tinted, xray_like


def test_accepts_grayscale_xray():
    assert rejection_reason(xray_like()) is None


def test_accepts_xray_saved_as_colour_file():
    gray = xray_like()
    assert rejection_reason(np.dstack([gray, gray, gray])) is None


def test_accepts_tinted_xray():
    assert rejection_reason(tinted(xray_like())) is None


def test_rejects_grayscale_noise():
    assert rejection_reason(noise()) == "noisy"


def test_rejects_colour_noise():
    assert rejection_reason(noise(channels=3)) is not None


def test_rejects_blank_images():
    assert rejection_reason(np.zeros((512, 512), np.uint8)) == "flat"
    assert rejection_reason(np.full((512, 512, 3), 255, np.uint8)) == "flat"


def test_rejects_colourful_image():
    assert rejection_reason(colourful()) == "colour"
