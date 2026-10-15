import struct
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from images import encode_png, noise, xray_like
from main import MAX_UPLOAD_BYTES, app

client = TestClient(app)


def make_png():
    return encode_png(xray_like())


def post_file(data, name="xray.png", content_type="image/png"):
    return client.post("/predict/", files={"file": (name, data, content_type)})


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_png():
    r = post_file(make_png())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "confidence", "pneumonia_probability"}

    p = body["pneumonia_probability"]
    assert 0.0 <= p <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    if p > 0.5:
        assert body["prediction"] == "Pneumonia Detected"
        assert body["confidence"] == pytest.approx(p)
    else:
        assert body["prediction"] == "Normal Lung"
        assert body["confidence"] == pytest.approx(1 - p)


def test_predict_garbage_bytes():
    r = post_file(b"this is not an image", name="x.png")
    assert r.status_code == 400
    assert r.json() == {"detail": "Could not read image. Please upload a JPG or PNG file."}


def test_predict_empty_file():
    r = post_file(b"")
    assert r.status_code == 400
    assert r.json() == {"detail": "Could not read image. Please upload a JPG or PNG file."}


def test_predict_too_large():
    r = post_file(b"\0" * (MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413
    assert r.json() == {"detail": "File too large. Maximum size is 10 MB."}


def test_predict_ignores_traversal_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = post_file(make_png(), name="../../evil.png")
    assert r.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_predict_rejects_noise():
    r = post_file(encode_png(noise()))
    assert r.status_code == 400
    assert r.json() == {
        "detail": "This doesn't look like a chest X-ray. Please upload a frontal chest X-ray image."
    }


def test_predict_rejects_huge_dimensions_in_small_file():
    # A blank 8000x8000 PNG is tiny on disk but 64 megapixels once decoded.
    data = encode_png(np.zeros((8000, 8000), np.uint8))
    assert len(data) < MAX_UPLOAD_BYTES
    r = post_file(data)
    assert r.status_code == 413
    assert r.json() == {"detail": "Image is too large. Maximum size is 50 megapixels."}


def test_predict_rejects_header_claiming_enormous_image():
    # Only the PNG header: claims 20000x20000 pixels with no real pixel data.
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", 20000, 20000, 8, 0, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")
    r = post_file(data)
    assert r.status_code == 413
