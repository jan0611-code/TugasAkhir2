import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
import time

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
# MODEL CONFIG
# =====================================================
MODEL_URL = "https://drive.google.com/file/d/1MH47HkGlOct_h458BgPBW2HzlTtZnW_3/view?usp=drive_link"
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = 224

# =====================================================
# LOAD MODEL & LABELS (BULLETPROOF VERSION)
# =====================================================
@st.cache_resource
def load_model():
    # Create labels.txt if it doesn't exist
    if not os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "w") as f:
            f.write("Phase_A\nPhase_B\nPhase_C")
    
    # Download model with multiple fallbacks
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model... This may take a minute.")
        
        # Try multiple download methods
        download_success = False
        
        # Method 1: requests with progress
        try:
            response = requests.get(MODEL_URL, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            if os.path.getsize(MODEL_PATH) > 1000:  # If file > 1KB
                download_success = True
                st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.warning(f"Method 1 failed: {e}")
        
        # Method 2: Direct from raw GitHub
        if not download_success:
            try:
                st.info("Trying alternative download method...")
                alt_url = "https://github.com/jan0611-code/TugasAkhir2/releases/download/v1.0/vgg16_trained_model.h5"
                r = requests.get(alt_url, timeout=30)
                with open(MODEL_PATH, 'wb') as f:
                    f.write(r.content)
                download_success = True
                st.success("✅ Model downloaded (method 2)!")
            except Exception as e:
                st.error(f"Method 2 failed: {e}")
        
        if not download_success:
            st.error("❌ Failed to download model. Please check:")
            st.write(f"1. URL: {MODEL_URL}")
            st.write("2. Internet connection")
            st.write("3. File exists on GitHub Releases")
            return None
    
    # Load the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        # Try to show file info
        if os.path.exists(MODEL_PATH):
            size = os.path.getsize(MODEL_PATH) / (1024*1024)
            st.write(f"Model file size: {size:.2f} MB")
        return None

@st.cache_data
def load_labels():
    try:
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except:
        return ["WD", "FWD", "GP"]

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
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter
    bilateral_gray = cv2.bilateralFilter(gray_img, d=11, sigmaColor=75, sigmaSpace=75)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        bilateral_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 9, 1
    )
    
    # Apply skeletonization
    skeleton_img = skeletonize(thresh)
    
    # Convert back to RGB for model input
    processed_rgb = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(processed_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Normalize and prepare for model
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Make prediction
    predictions = model.predict(input_tensor, verbose=0)[0]
    idx = np.argmax(predictions)
    
    return img_rgb, skeleton_img, class_names[idx], predictions[idx], predictions

# =====================================================
# MAIN APP LOGIC
# =====================================================
# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("App cannot start without model. Please check the logs.")
    st.stop()

# Load labels
class_names = load_labels()

# UI Elements
st.write(f"Model loaded successfully! Ready to classify {len(class_names)} phases.")

input_method = st.radio("Select input method:", ["Upload Image", "Camera"])
image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.success("✅ Image uploaded!")
else:
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image = Image.open(camera_file).convert("RGB")
        st.success("✅ Image captured!")

# Process image
if image is not None:
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("🔍 Classify Now", type="primary"):
        with st.spinner("Processing image..."):
            original, skeleton, label, confidence, all_preds = preprocess_and_predict(image)
        
        st.success("✅ Classification complete!")
        
        # Show results
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original", use_column_width=True)
        with col2:
            st.image(skeleton, caption="Skeletonized", use_column_width=True)
        
        # Show prediction
        st.markdown(f"### 📊 Prediction: **{label}**")
        st.markdown(f"#### Confidence: **{confidence*100:.1f}%**")
        
        # Show all scores
        st.markdown("### 📈 All Class Scores:")
        for name, score in zip(class_names, all_preds):
            st.progress(float(score), text=f"{name}: {score*100:.2f}%")

# =====================================================
# DEBUG INFO (hidden by default)
# =====================================================
with st.expander("🔧 Debug Info"):
    st.write("Model info:")
    if model:
        st.write(f"- Input shape: {model.input_shape}")
        st.write(f"- Output shape: {model.output_shape}")
        st.write(f"- Number of classes: {len(class_names)}")
    
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH) / (1024*1024)
        st.write(f"- Model file size: {size:.2f} MB")
    
    st.write(f"- Labels: {class_names}")
    st.write(f"- Python version: 3.10.19")
    st.write(f"- TensorFlow version: {tf.__version__}")

