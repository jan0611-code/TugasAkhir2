import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import urllib.request
import tempfile
import io

# Page configuration
st.set_page_config(
    page_title="NLC Phase Classifier",
    page_icon="☁️",
    layout="wide"
)

# Title and description
st.title("☁️ Noctilucent Cloud (NLC) Phase Classification System")
st.markdown("Upload an NLC image or use your camera to automatically classify its phase")

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ----------------------------
# EXACT FUNCTIONS FROM GITHUB
# ----------------------------
def skeletonize(img):
    """Skeletonization function from GitHub prediction.py"""
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

def preprocess_image_pil(image_pil, img_size=224):
    """Adapted preprocessing from GitHub for PIL Image input"""
    # Convert PIL Image to OpenCV format
    img_array = np.array(image_pil)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Convert to RGB (like original GitHub code)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # EXACT preprocessing from GitHub
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)

    thresh = cv2.adaptiveThreshold(
        bilateral, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9, 1
    )

    skeleton = skeletonize(thresh)

    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(skeleton_rgb, (img_size, img_size))

    # Prepare for model
    x = resized.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    return x, skeleton_rgb

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    ### Noctilucent Cloud Phase Classification Model
    
    **Model Features:**
    - Uses VGG16 architecture (from GitHub)
    - Preprocessing: Bilateral filter + adaptive threshold + skeletonization
    - **3 Phases:** phase1, phase2, phase3
    
    **Model Source:** [jan0611-code/TugasAkhir2](https://github.com/jan0611-code/TugasAkhir2)
    """)
    
    # Add preprocessing visualization toggle
    show_preprocessing = st.checkbox("Show preprocessing steps", value=True)
    
    st.markdown("---")
    st.subheader("Model Management")
    
    # Model download and loading - EXACT from GitHub
    @st.cache_resource
    def download_and_load_model():
        """Download and load the model - using GitHub's model"""
        try:
            # Model download URL from GitHub releases
            model_url = "https://github.com/jan0611-code/TugasAkhir2/releases/download/v1.0/model.h5"
            
            # Create temporary file for model
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                model_path = tmp_file.name
            
            # Download model
            with st.spinner("Downloading model file..."):
                urllib.request.urlretrieve(model_url, model_path)
            
            # Load model using TensorFlow (like GitHub)
            with st.spinner("Loading VGG16 model..."):
                model = tf.keras.models.load_model(model_path)
            
            # Clean up temporary file
            os.unlink(model_path)
            
            st.success("✅ Model architecture loaded")
            st.write(f"Input shape: {model.input_shape}")
            st.write(f"Output shape: {model.output_shape}")
            
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    if st.button("Download & Load Model", type="primary"):
        with st.spinner("Processing..."):
            model = download_and_load_model()
            if model:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🖼️ Image Input")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Upload Image", "Use Camera"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Image upload option
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an NLC image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                st.session_state.camera_image = None
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.success(f"✅ Image uploaded successfully! Size: {image.size} | Mode: {image.mode}")
            except Exception as e:
                st.error(f"❌ Failed to read image: {e}")
    
    # Camera option
    else:
        st.markdown("**Use camera to capture NLC image**")
        camera_image = st.camera_input("Click to take photo", label_visibility="collapsed")
        
        if camera_image is not None:
            try:
                image = Image.open(camera_image)
                st.session_state.camera_image = image
                st.session_state.uploaded_image = None
                
                st.image(image, caption="Captured Image", use_column_width=True)
                st.success("✅ Photo captured successfully!")
            except Exception as e:
                st.error(f"❌ Failed to process image: {e}")

with col2:
    st.header("📈 Classification Results")
    
    # Create result placeholders
    result_container = st.container()
    preprocessing_container = st.container()
    confidence_container = st.container()
    details_container = st.container()
    
    # Get current image
    current_image = None
    if st.session_state.uploaded_image:
        current_image = st.session_state.uploaded_image
    elif st.session_state.camera_image:
        current_image = st.session_state.camera_image
    
    # Display current status
    if not st.session_state.model_loaded:
        result_container.warning("⚠️ Please load the model first in the sidebar")
    
    elif current_image is not None:
        result_container.info("📷 Image ready, click button below to start classification")
        
        # Classification button
        if st.button("🚀 Begin Classification", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # ----------------------------
                    # EXACT PREPROCESSING FROM GITHUB
                    # ----------------------------
                    processed_image, skeleton_img = preprocess_image_pil(current_image)
                    
                    # Show preprocessing steps if enabled
                    if show_preprocessing:
                        with preprocessing_container:
                            st.markdown("---")
                            st.subheader("🔬 Preprocessing Steps")
                            
                            # Convert to numpy for display
                            original_array = np.array(current_image)
                            
                            # Apply the same steps for visualization
                            if len(original_array.shape) == 3:
                                img_rgb = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
                                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                            else:
                                img_rgb = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
                            
                            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                            bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
                            thresh = cv2.adaptiveThreshold(
                                bilateral, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                9, 1
                            )
                            
                            # Display steps in columns
                            col1_pre, col2_pre, col3_pre, col4_pre = st.columns(4)
                            
                            with col1_pre:
                                st.image(original_array, caption="Original", use_column_width=True)
                            with col2_pre:
                                st.image(bilateral, caption="Bilateral Filter", use_column_width=True, clamp=True)
                            with col3_pre:
                                st.image(thresh, caption="Adaptive Threshold", use_column_width=True, clamp=True)
                            with col4_pre:
                                st.image(skeleton_img, caption="Skeletonized", use_column_width=True, clamp=True)
                    
                    # ----------------------------
                    # MAKE PREDICTION (like GitHub)
                    # ----------------------------
                    predictions = st.session_state.model.predict(processed_image, verbose=0)
                    
                    # Get prediction results
                    predicted_class = np.argmax(predictions[0])
                    confidence_scores = predictions[0]
                    
                    # EXACT CLASS NAMES FROM GITHUB
                    phase_labels = {
                        0: "Phase 1",
                        1: "Phase 2", 
                        2: "Phase 3"
                    }
                    
                    phase_descriptions = {
                        0: "Early formation stage with sparse structures",
                        1: "Mature development stage with clear veils", 
                        2: "Advanced or dissipating stage with complex patterns"
                    }
                    
                    # Display main results
                    with result_container:
                        st.markdown("---")
                        st.subheader("🎯 Classification Result")
                        
                        # Display predicted phase
                        predicted_phase = phase_labels.get(predicted_class, f"Phase {predicted_class + 1}")
                        top_confidence = confidence_scores[predicted_class] * 100
                        
                        # Color coding
                        if top_confidence > 80:
                            confidence_color = "#4CAF50"  # Green
                            emoji = "✅"
                        elif top_confidence > 60:
                            confidence_color = "#FF9800"  # Orange
                            emoji = "⚠️"
                        else:
                            confidence_color = "#F44336"  # Red
                            emoji = "❓"
                        
                        col_result1, col_result2 = st.columns(2)
                        with col_result1:
                            st.markdown(f"""
                            <div style="padding: 15px; border-radius: 10px; border-left: 5px solid {confidence_color}; background-color: #f8f9fa;">
                                <h3 style="margin: 0; color: #333;">Predicted Phase</h3>
                                <h1 style="margin: 10px 0; color: {confidence_color};">{predicted_phase}</h1>
                                <p style="margin: 0; color: #666;">{emoji} {phase_descriptions.get(predicted_class, '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_result2:
                            st.markdown(f"""
                            <div style="padding: 15px; border-radius: 10px; border-left: 5px solid {confidence_color}; background-color: #f8f9fa;">
                                <h3 style="margin: 0; color: #333;">Confidence Score</h3>
                                <h1 style="margin: 10px 0; color: {confidence_color};">{top_confidence:.1f}%</h1>
                                <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; margin: 10px 0;">
                                    <div style="background-color: {confidence_color}; width: {top_confidence}%; border-radius: 10px; height: 100%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display confidence distribution
                    with confidence_container:
                        st.markdown("---")
                        st.subheader("📊 Confidence Distribution")
                        
                        # Create progress bars for each phase
                        for i in range(len(confidence_scores)):
                            phase_name = phase_labels.get(i, f"Phase {i + 1}")
                            confidence = confidence_scores[i] * 100
                            
                            # Create column layout
                            col_label, col_progress, col_value = st.columns([2, 5, 1])
                            
                            with col_label:
                                is_current = (i == predicted_class)
                                label_icon = "🎯" if is_current else "○"
                                st.write(f"{label_icon} **{phase_name}**")
                            
                            with col_progress:
                                # Set color based on confidence
                                if confidence > 70:
                                    color = "#4CAF50"
                                elif confidence > 40:
                                    color = "#FF9800"
                                else:
                                    color = "#F44336"
                                
                                # Custom progress bar style
                                st.markdown(f"""
                                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 2px;">
                                    <div style="background-color: {color}; width: {confidence}%; border-radius: 8px; padding: 5px;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_value:
                                st.write(f"**{confidence:.1f}%**")
                    
                    # Display detailed information
                    with details_container:
                        st.markdown("---")
                        with st.expander("🔍 View Detailed Analysis"):
                            # Image information
                            st.markdown("### Model Information")
                            st.write(f"**Model Architecture:** VGG16 (from GitHub)")
                            st.write(f"**Input Shape:** {processed_image.shape}")
                            st.write(f"**Number of Classes:** 3")
                            
                            # Prediction statistics
                            st.markdown("### Prediction Statistics")
                            max_confidence = np.max(confidence_scores) * 100
                            min_confidence = np.min(confidence_scores) * 100
                            mean_confidence = np.mean(confidence_scores) * 100
                            std_confidence = np.std(confidence_scores) * 100
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            with col_stat1:
                                st.metric("Highest", f"{max_confidence:.1f}%")
                            with col_stat2:
                                st.metric("Lowest", f"{min_confidence:.1f}%")
                            with col_stat3:
                                st.metric("Average", f"{mean_confidence:.1f}%")
                            with col_stat4:
                                st.metric("Std Dev", f"{std_confidence:.1f}%")
                            
                            # Raw prediction values
                            st.markdown("### Raw Prediction Values")
                            for i, score in enumerate(confidence_scores):
                                confidence_percent = score * 100
                                st.code(f"Phase {i + 1}: {score:.6f} ({confidence_percent:.2f}%)")
                            
                            # Recommendation
                            st.markdown("### Interpretation")
                            if top_confidence > 80:
                                st.success("High confidence - Result is highly reliable")
                            elif top_confidence > 60:
                                st.info("Moderate confidence - Result is likely accurate")
                            else:
                                st.warning("Low confidence - Consider retaking the image with clearer NLC features")
                
                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
                    st.exception(e)
    
    else:
        result_container.info("👈 Please upload an image or use the camera first")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Noctilucent Cloud (NLC) Phase Classification System v1.0</p>
    <p>Using exact preprocessing from: <a href="https://github.com/jan0611-code/TugasAkhir2">jan0611-code/TugasAkhir2</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .metric-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
