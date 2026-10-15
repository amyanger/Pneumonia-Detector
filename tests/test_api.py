import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import MAX_UPLOAD_BYTES, app

client = TestClient(app)


def make_png(size=(256, 256)):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=size, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


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
