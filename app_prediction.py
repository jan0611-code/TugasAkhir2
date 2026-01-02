import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import requests

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="NLC Phase Classification",
    layout="centered"
)

st.title("NLC Phase Classification")
st.write("Upload an image or use the camera, then classify the NLC phase.")

# =====================================================
# MODEL CONFIG (GitHub Release)
# =====================================================
MODEL_URL = "https://github.com/jan0611-code/TugasAkhir2/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = 224

# =====================================================
# LOAD MODEL & LABELS
# =====================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)..."):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(MODEL_URL, headers=headers, stream=True)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                st.success("✓ Model downloaded")
                
            except Exception as e:
                st.error(f"✗ Download failed: {e}")
                return None
    
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"✗ Model loading failed: {e}")
        return None

@st.cache_data
def load_labels():
    try:
        with open(LABELS_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    except:
        return ["Class 1", "Class 2", "Class 3"]

# Load
model = load_model()
class_names = load_labels()

if model is None:
    st.error("Cannot start app without model.")
    st.stop()

# =====================================================
# SKELETONIZATION FUNCTION
# =====================================================
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

# =====================================================
# PREPROCESS & PREDICT
# =====================================================
def preprocess_and_predict(pil_image):
    img_rgb = np.array(pil_image)
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bilateral_gray = cv2.bilateralFilter(gray_img, d=11, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(bilateral_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)
    skeleton_img = skeletonize(thresh)
    processed_rgb = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(processed_rgb, (IMG_SIZE, IMG_SIZE))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    predictions = model.predict(input_tensor, verbose=0)[0]
    idx = np.argmax(predictions)
    return img_rgb, skeleton_img, class_names[idx], predictions[idx], predictions

# =====================================================
# UI
# =====================================================
input_method = st.radio("Select input method:", ["Upload Image", "Camera"])
image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image = Image.open(camera_file).convert("RGB")

if image is not None:
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Begin Classification", type="primary"):
        with st.spinner("Processing..."):
            original, skeleton, label, confidence, all_preds = preprocess_and_predict(image)
        
        st.success("Done!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original", use_column_width=True)
        with col2:
            st.image(skeleton, caption="Skeletonized", use_column_width=True, clamp=True)
        
        st.markdown(f"**Predicted Phase:** {label}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
        
        st.subheader("All predictions:")
        for name, score in zip(class_names, all_preds):
            st.progress(score, text=f"{name}: {score * 100:.2f}%")
