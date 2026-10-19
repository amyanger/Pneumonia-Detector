# Pneumonia Detector

A small deep learning project that classifies chest X-rays as normal or pneumonia. It includes a training pipeline for the Kaggle chest X-ray dataset, a FastAPI backend that serves predictions, and a plain HTML/JS frontend for uploading an image.

This is an educational project. It is not a medical device and should not be used for diagnosis.

## Project layout

```
Pneumonia-Detector/
├── backend/
│   ├── main.py                          FastAPI app
│   ├── image_checks.py                  rejects uploads that are clearly not X-rays
│   └── pneumonia_detector_model.keras   trained model (tracked in git)
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── tests/
│   ├── conftest.py
│   ├── images.py                        synthetic test images
│   ├── test_api.py                      API tests
│   └── test_image_checks.py             tests for the X-ray checks
├── extract_data.py                      unzip the Kaggle dataset
├── load_images.py                       show one sample image per class
├── preprocess_data.py                   build train/validation arrays
├── train_model.py                       train and save the model
├── evaluate_model.py                    evaluate on the held-out test set
├── test_model.py                        predict a single image
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

## Setup

You need Python 3.13. TensorFlow does not support Python 3.14 yet, so if `python3` on your machine is 3.14, call `python3.13` explicitly.

```sh
git clone https://github.com/amyanger/Pneumonia-Detector.git
cd Pneumonia-Detector
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned exactly because the bundled model was saved with Keras 3.8 and does not load on Keras 3.11 or newer, which in turn holds TensorFlow at 2.20.0.

## Quick start

The repo ships with a trained model, so you can run the app without downloading the dataset.

1. Start the backend (it works from any directory):

   ```sh
   python backend/main.py
   ```

   It serves on http://127.0.0.1:8000. Interactive API docs are at http://127.0.0.1:8000/docs.

2. Open `frontend/index.html` directly in Safari while the backend is running. The page sends uploads to `http://localhost:8000`.

3. Upload a chest X-ray (JPG or PNG, up to 10 MB) to get a prediction.

## Training pipeline

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. Other chest X-ray datasets use different formats and labels and are not supported.

Run these from the repo root with the venv active:

```sh
python extract_data.py path/to/archive.zip   # defaults to archive.zip
python load_images.py                        # optional, shows one sample per class
python preprocess_data.py
python train_model.py                        # --epochs N, default 20
python evaluate_model.py                     # --threshold, default 0.5
python test_model.py dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg
```

What each step does:

- `extract_data.py` unpacks the zip into `dataset/chest_xray/{train,test,val}`. It handles the nested duplicate folder and `__MACOSX` entries found in some copies of the zip.
- `preprocess_data.py` loads `train/`, converts each image to grayscale 224x224 scaled to [0, 1], and makes an 80/20 train/validation split grouped by patient, so no patient appears in both sets. It writes `X_train.npy`, `X_val.npy`, `y_train.npy` and `y_val.npy`.
- `train_model.py` trains with class weights to offset the class imbalance, early stopping, and learning rate reduction on plateau. It saves to `backend/pneumonia_detector_model.keras`, or to the path in the `MODEL_PATH` environment variable if set. A retrained model is saved in whatever Keras version is installed.
- `evaluate_model.py` runs the model on the held-out `test/` set and prints a classification report, confusion matrix and ROC AUC.
- `test_model.py` predicts a single image and displays it with the result.

The dataset and `.npy` paths are relative to your current directory, which is why the commands should be run from the repo root. The model path is resolved relative to the scripts themselves.

### Model

Input is a grayscale 224x224 image scaled to [0, 1]. A 3x3 convolution maps the single channel to three, which feeds a frozen MobileNetV2 pretrained on ImageNet. On top of that sit global average pooling, a 128-unit dense layer, dropout, and a sigmoid output. An output above 0.5 means pneumonia.

## API

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /predict/`

Multipart form upload with a single field named `file` (JPG or PNG, max 10 MB).

```sh
curl -F "file=@path/to/xray.jpeg" http://127.0.0.1:8000/predict/
```

Response:

```json
{
  "prediction": "Pneumonia Detected",
  "confidence": 0.93,
  "pneumonia_probability": 0.93
}
```

`prediction` is either `"Pneumonia Detected"` or `"Normal Lung"`. `confidence` is the probability of the predicted label, so for a normal result it equals `1 - pneumonia_probability`.

Errors return a JSON body of the form `{"detail": "..."}`:

- 400 if the file is empty or cannot be decoded as an image
- 400 if the image is clearly not a chest X-ray (see below)
- 413 if the file is larger than 10 MB, or the image is larger than 50 megapixels
- 500 if the model fails during prediction

### Rejecting non-X-ray images

The model gives a confident answer for any input, so before predicting, the backend runs three quick checks in `backend/image_checks.py` and rejects images that are:

- nearly blank (very little contrast)
- noisy (neighbouring pixels jump around far more than in a real X-ray)
- clearly multicoloured (a tinted X-ray, such as a photo of film on a lightbox, still passes)

The cutoffs were set on 33 public chest X-rays from Wikimedia Commons, all of which pass, plus noise, blank images and colour photos. Random noise and blank images are always rejected, and about 60% of the colour photos tried were. Muted or black and white photos and X-rays of other body parts still get through.

## Running tests

```sh
pip install -r requirements-dev.txt
pytest tests -q
```

The API tests use the real bundled model, so TensorFlow has to be installed. Test images are generated synthetically, so no dataset is needed.

## Limitations

- No accuracy figures are published here. The bundled model has not been measured on the test set in this repo. Run `evaluate_model.py` to get real numbers.
- The model itself has no idea whether an image is a chest X-ray. The checks above stop obvious junk like noise, blank images and colourful photos, but a grayscale photo or an X-ray of another body part will still get a confident answer. Catching those would need a learned check fitted on the training data.
- The Kaggle dataset contains only pediatric chest X-rays from a single source, Guangzhou Women and Children's Medical Center. Results on adults or on images from other hospitals and equipment are not reliable.

## Author

amyanger - [GitHub](https://github.com/amyanger)

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
