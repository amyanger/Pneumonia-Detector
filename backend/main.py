import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
import cv2

from image_checks import rejection_reason

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "pneumonia_detector_model.keras"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI()

# Origins stay "*" because the frontend is often opened via file:// (origin "null").
# No cookies or auth are used, so credentials are disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(MODEL_PATH)


def decode_image(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for CNN
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.get("/health")
def health():
    return {"status": "ok"}


# Sync route on purpose: FastAPI runs it in a threadpool so model.predict
# does not block the event loop.
@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    img = decode_image(data) if data else None
    if img is None:
        raise HTTPException(
            status_code=400, detail="Could not read image. Please upload a JPG or PNG file."
        )

    reason = rejection_reason(img)
    if reason:
        logger.info("Rejected upload as not a chest X-ray (%s)", reason)
        raise HTTPException(
            status_code=400,
            detail="This doesn't look like a chest X-ray. Please upload a frontal chest X-ray image.",
        )

    try:
        probability = float(model.predict(preprocess_image(img), verbose=0)[0][0])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    is_pneumonia = probability > 0.5
    return {
        "prediction": "Pneumonia Detected" if is_pneumonia else "Normal Lung",
        "confidence": probability if is_pneumonia else 1 - probability,
        "pneumonia_probability": probability,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
