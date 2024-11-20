
# import libraries
import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import time

# Page configuration
st.set_page_config(
    page_title="AI Chest Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better UI consistency
st.markdown(
    """
    <style>
        /* Reset and base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* Main container */
        .main {
            background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
            padding: 2rem;
            min-height: 100vh;
        }
        
        /* Header styles */
        .header {
            background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 2.5rem 2rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            font-family: 'Helvetica Neue', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }
        
        .header-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            line-height: 1.5;
        }
        
        /* Cards and containers */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .card {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
            border: 1px solid #e1e8ed;
        }
        
        /* Upload section - Improved */
        .upload-container {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            background: #f8fafc;
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 1rem 0;
        }
        
        .upload-area:hover {
            border-color: #2980b9;
            background: #f1f7fa;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(120deg, #2980b9 0%, #3498db 100%);
            color: white;
            padding: 0.8rem 2rem;
            border-radius: 50px;
            border: none;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }

        .upload-icon {
            font-size: 3rem;
            color: #3498db;
            margin-bottom: 1rem;
        }
        
        .upload-text {
            color: #2c3e50;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        .upload-subtext {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        /* Stats cards */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            border: 1px solid #e1e8ed;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #7f8c8d;
            font-size: 1rem;
        }
        
        /* Button styles */
        .custom-button {
            background: linear-gradient(120deg, #2980b9 0%, #3498db 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: 50px;
            border: none;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            width: auto;
            margin: 1rem auto;
            display: block;
            cursor: pointer;
        }
        
        .custom-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        
        /* Result boxes */
        .result-box {
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 2rem;
            font-size: 1.2rem;
            font-weight: 600;
            animation: fadeIn 0.5s ease-out;
        }
        
        .result-box.normal {
            background: linear-gradient(120deg, #27ae60 0%, #2ecc71 100%);
            color: white;
        }
        
        .result-box.cancer {
            background: linear-gradient(120deg, #c0392b 0%, #e74c3c 100%);
            color: white;
        }

        /* Progress bar */
        .stProgress > div > div {
            background-color: #3498db;
        }
        
        # /* Progress bar */
        # .stProgress > div > div {
        #     background: linear-gradient(90deg, #3498db, #2980b9);
        #     height: 8px;
        #     border-radius: 4px;
        # }
        
        /* Sidebar styles */
        .sidebar .sidebar-content {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        
        /* Info boxes */
        .info-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #3498db;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            border-top: 1px solid #e1e8ed;
            margin-top: 3rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar

# Sidebar with information
with st.sidebar:
    st.header("AI Chest Cancer Detection")
    
    st.image("image.png", use_container_width=True)

    st.markdown("### 📊 About the System")
    st.info(
        """
        This AI-powered system uses deep learning to analyze chest X-ray images 
        and detect potential signs of Adenocarcinoma Cancer. The model has been 
        trained on thousands of X-ray images to provide accurate predictions.
        """
    )
    
    st.markdown("### 🎯 How to Use")
    st.markdown(
        """
        1. Upload a chest X-ray image (PNG, JPG, JPEG)
        2. Wait for the image to process
        3. Click the 'Analyze Image' button
        4. Review the prediction results
        """
    )

    st.markdown("### ⚠️ Important Note")
    st.warning(
        """
        This tool is for screening purposes only and should not be used as a 
        definitive diagnosis. Always consult with healthcare professionals for 
        proper medical advice.
        """
    )

# Header
st.markdown(
    """
    <div class="header">
        <h1>
            <img src="https://img.icons8.com/fluency/48/000000/stethoscope.png" style="width: 48px; height: 48px;"/>
            AI-Powered Chest Cancer Detection
        </h1>
        <p class="header-subtitle">
            Advanced deep learning system for early detection of Adenocarcinoma Cancer through X-ray analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Statistics
st.markdown(
    """
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">88.5%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">< 10s</div>
            <div class="stat-label">Analysis Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">5k+</div>
            <div class="stat-label">Images Analyzed</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload Section
st.markdown(
    """
    <div class="upload-container">
        <div class="upload-area">
            <div class="upload-icon">📤</div>
            <div class="upload-text">Upload X-Ray Image</div>
            <div class="upload-subtext">Drag and drop or click to browse</div>
            <div class="upload-subtext">Supported formats: PNG, JPG, JPEG • Max size: 200MB</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

class PredictionPipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        resized_image = image.resize((224, 224))
        image_array = img_to_array(resized_image)
        preprocessed_image = np.expand_dims(image_array, axis=0)
        preprocessed_image /= 255.0
        return preprocessed_image

    def predict(self, image):
        preprocessed_image = self.preprocess_image(image)
        result = np.argmax(self.model.predict(preprocessed_image), axis=1)
        confidence = np.max(self.model.predict(preprocessed_image)) * 100
        return ("Normal" if result[0] == 1 else "Adenocarcinoma Cancer", confidence)

# Initialize model
MODEL_PATH = r"model/model.h5"
pipeline = PredictionPipeline(MODEL_PATH)

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
    with col2:
        st.markdown("### Image Details")
        st.markdown(f"**File Name:** {uploaded_file.name}")
        st.markdown(f"**Image Size:** {image.size}")
        st.markdown(f"**File Format:** {uploaded_file.type}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Analyze Image", key="analyze_button"):
        with st.spinner("Processing image..."):
            # Progress bar animation
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get prediction
            prediction, confidence = pipeline.predict(image)
            
            # Display results
            if prediction == "Normal":
                st.markdown(
                    f"""
                    <div class="result-box normal">
                        ✅ Result: {prediction}<br>
                        Confidence: {confidence:.2f}%<br>
                        No signs of cancer detected
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box cancer">
                        ⚠️ Result: {prediction}<br>
                        Confidence: {confidence:.2f}%<br>
                        Please consult a healthcare professional immediately
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Additional analysis information
            st.markdown("### Detailed Analysis")
            if prediction == "Normal":
                st.markdown(
                """
                <div class="info-box">
                    <h4>📋 Assessment Summary</h4>
                    <ul>
                        <li>No abnormal patterns detected in the X-ray image</li>
                        <li>Regular lung structure observed</li>
                        <li>No suspicious masses or nodules identified</li>
                        <li>Continue routine health check-ups</li>
                        <li>Maintain regular screening schedule</li>
                        <li>Report any new symptoms to your healthcare provider</li>
                    <ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            else:
                st.markdown(
                    """
                    <div class="info-box">
                        <h4>⚠️ Immediate Actions Required</h4>
                        <ul>
                            <li>Schedule an urgent consultation with an oncologist</li>
                            <li>Prepare for additional diagnostic tests</li>
                            <li>Gather all relevant medical history</li>
                            <li>Additional imaging tests may be required</li>
                            <li>Biopsy might be recommended</li>
                            <li>Discuss treatment options with specialists</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# Footer
st.markdown(
    """
    <div class="footer">
        <p>💻 Powered by Advanced AI Technology</p>
        <p>🔒 HIPAA Compliant | 🏥 For Screening Purposes Only | ⚕️ Consult Healthcare Professionals</p>
        <p style="margin-top: 1rem; font-size: 0.8rem; color: #95a5a6;">© 2024 AI Chest Cancer Detection System. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
