import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import requests  # Added for better download handling

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
MODEL_URL = (
    "https://github.com/jan0611-code/TugasAkhir2/"
    "releases/download/v1.0/model.h5"
)
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
            # Use requests for reliable download with headers
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(MODEL_URL, headers=headers, stream=True)
                response.raise_for_status()  # Check for HTTP errors
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(MODEL_PATH, 'wb') as f:
                    if total_size == 0:  # No content length header
                        f.write(response.content)
                    else:
                        downloaded = 0
                        progress_bar = st.progress(0)
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                progress_bar.progress(min(downloaded / total_size, 1.0))
                        progress_bar.empty()
                
                st.success("Model downloaded successfully!")
                
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.info("Please check if the model file exists at the URL.")
                return None
    
    # Load the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data
def load_labels():
    try:
        with open(LABELS_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Failed to load labels: {e}")
        return ["Class 1", "Class 2", "Class 3"]  # Fallback labels

# Load model and labels
model = load_model()
class_names = load_labels()

# Check if model loaded successfully
if model is None:
    st.error("Failed to load model. Please check the model file and try again.")
    st.stop()

# =====================================================
# SKELETONIZATION FUNCTION (UNCHANGED LOGIC)
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
    # Convert PIL image to numpy array
    img_rgb = np.array(pil_image)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter
    bilateral_gray = cv2.bilateralFilter(
        gray_img, d=11, sigmaColor=75, sigmaSpace=75
    )

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        bilateral_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9,
        1
    )

    # Apply skeletonization
    skeleton_img = skeletonize(thresh)

    # Convert back to RGB for model input
    processed_rgb = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2RGB)
    
    # Resize to model input size
    resized = cv2.resize(processed_rgb, (IMG_SIZE, IMG_SIZE))

    # Normalize and prepare for model
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Make prediction
    predictions = model.predict(input_tensor, verbose=0)[0]
    idx = np.argmax(predictions)

    return img_rgb, skeleton_img, class_names[idx], predictions[idx], predictions

# =====================================================
# IMAGE INPUT
# =====================================================
input_method = st.radio("Select input method:", ["Upload Image", "Camera"])

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.success("Image uploaded successfully!")

else:
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image = Image.open(camera_file).convert("RGB")
        st.success("Image captured successfully!")

# =====================================================
# DISPLAY & CLASSIFY
# =====================================================
if image is not None:
    # Display original image
    st.subheader("Input Image")
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Begin Classification", type="primary"):
        with st.spinner("Processing & classifying..."):
            original, skeleton, label, confidence, all_preds = (
                preprocess_and_predict(image)
            )

        st.success("Classification complete!")

        # Display original and processed images side by side
        st.subheader("Processing Results")
        col1, col2 = st.columns(2)

        with col1:
            st.image(original, caption="Original Image", use_column_width=True)

        with col2:
            st.image(
                skeleton,
                caption="Processed Image (Skeletonized)",
                use_column_width=True,
                clamp=True
            )

        # Display prediction results
        st.subheader("📊 Classification Results")
        
        # Create a nice result box
        result_container = st.container()
        with result_container:
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin: 10px 0;
            ">
                <h3 style="margin-top: 0;">Predicted Phase: <strong>{label}</strong></h3>
                <p style="font-size: 1.2em;">Confidence: <strong>{confidence * 100:.2f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # Display confidence scores for all classes
        st.subheader("📈 Confidence Scores per Class")
        
        # Create a progress bar for each class
        for name, score in zip(class_names, all_preds):
            percentage = score * 100
            color = "#4CAF50" if percentage > 70 else "#FF9800" if percentage > 30 else "#f44336"
            
            st.markdown(f"**{name}**")
            st.progress(score, text=f"{percentage:.2f}%")

        # Optional: Show raw prediction values
        with st.expander("Show raw prediction values"):
            for name, score in zip(class_names, all_preds):
                st.write(f"{name}: {score:.6f}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "NLC Phase Classification App | Built with Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
