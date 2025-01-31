from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # ✅ Import CORS Middleware
import uvicorn
import numpy as np
import tensorflow as tf
import cv2
import os

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# ✅ Load the trained model
model = tf.keras.models.load_model("pneumonia_detector_model.keras")


# ✅ Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for CNN
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# ✅ Endpoint to receive and predict uploaded image
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        file_path = f"temp_{file.filename}"

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        img = preprocess_image(file_path)
        if img is None:
            return {"error": "Invalid image format"}

        prediction = model.predict(img)[0][0]
        os.remove(file_path)

        result = "Pneumonia Detected" if prediction > 0.5 else "Normal Lung"
        return {"prediction": result, "confidence": float(prediction)}

    except Exception as e:
        return {"error": str(e)}


# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
