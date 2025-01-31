import numpy as np
import tensorflow as tf
import cv2
import sys
import os
import matplotlib.pyplot as plt  # ✅ Added for visualization

# ✅ Load the trained model
model = tf.keras.models.load_model("pneumonia_detector_model.keras")


# ✅ Function to preprocess image
def preprocess_image(image_path):
    if not os.path.exists(image_path):  # Check if file exists
        print(f"❌ Error: Image file not found at {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    if img is None:
        print(f"❌ Error: OpenCV could not read the image file at {image_path}")
        sys.exit(1)

    img = cv2.resize(img, (224, 224))  # Resize to match model input shape
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for CNN
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# ✅ Predict Function (Now Includes Image Display)
def predict_pneumonia(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]  # Get prediction

    # ✅ Determine Prediction Label
    if prediction > 0.5:
        label = f"🔴 Pneumonia Detected (Confidence: {prediction:.2f})"
    else:
        label = f"🟢 Normal Lung (Confidence: {1 - prediction:.2f})"

    # ✅ Show Image with Prediction
    img_display = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load for display
    plt.imshow(img_display, cmap="gray")
    plt.title(label)
    plt.axis("off")
    plt.show()

    # ✅ Print Prediction in Console Too
    print(label)


# ✅ Test the model on a sample image
if len(sys.argv) > 1:
    test_image = sys.argv[1]  # Pass image path as argument
    predict_pneumonia(test_image)
else:
    print("⚠️ Please provide an image path. Example:")
    print(
        "   python test_model.py dataset/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    )
