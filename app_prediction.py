import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os

# =========================
# CONFIG & MODEL DOWNLOAD
# =========================
IMG_SIZE = 224
CLASS_NAMES = ["WD", "FWD", "GP"]
MODEL_PATH = "vgg16_trained_model.h5"
# Extracted ID from your Google Drive link
GDRIVE_ID = "1MH47HkGlOct_h458BgPBW2HzlTtZnW_3"

@st.cache_resource
def load_vgg_model():
    # Check if model exists, if not, download it
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... This may take a minute."):
            url = f'https://drive.google.com/uc?id={GDRIVE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return load_model(MODEL_PATH)

# Initialize model
model = load_vgg_model()

# =========================
# SKELETONIZATION FUNCTION
# =========================
def skeletonize(img):
    if img.max() > 1:
        img = img // 255

    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break
    return skel

# =========================
# PREPROCESSING PIPELINE
# =========================
def preprocess_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)

    thresh = cv2.adaptiveThreshold(
        bilateral, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9, 1
    )

    skel = skeletonize(thresh)
    skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(skel_rgb, (IMG_SIZE, IMG_SIZE))

    model_input = resized.astype("float32") / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    return img_rgb, gray, bilateral, thresh, skel, model_input

# =========================
# UI
# =========================
st.set_page_config(page_title="Phase Classification", layout="centered")
st.title("🔬 Phase Classification")
st.write("Upload an image or use your camera. The system applies **OpenCV preprocessing** before classification.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

camera_file = st.camera_input("Or take a photo")
input_file = uploaded_file if uploaded_file else camera_file

# =========================
# PROCESS & PREDICT
# =========================
if input_file is not None:
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Failed to read image.")
    else:
        (img_rgb, gray, bilateral, thresh, skel, model_input) = preprocess_image(img_bgr)

        st.subheader("🧪 Preprocessing Stages")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)
            st.image(gray, caption="Grayscale", use_container_width=True)

        with col2:
            st.image(bilateral, caption="Bilateral Filter", use_container_width=True)
            st.image(thresh, caption="Adaptive Threshold", use_container_width=True)

        with col3:
            st.image(skel, caption="Skeletonized", use_container_width=True)

        if st.button("🚀 Begin Classify"):
            preds = model.predict(model_input)[0]
            idx = np.argmax(preds)

            st.subheader("📊 Prediction Result")
            st.success(f"**Predicted Phase:** {CLASS_NAMES[idx]}")
            st.write(f"**Confidence:** {preds[idx]*100:.2f}%")

            st.write("### All Class Probabilities")
            for i, cls in enumerate(CLASS_NAMES):
                st.write(f"{cls}: {preds[i]*100:.2f}%")
