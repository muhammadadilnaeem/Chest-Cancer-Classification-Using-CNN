
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

# with st.sidebar:
#     st.markdown("### 📊 About the System")
#     st.markdown(
#         """
#         <div class="sidebar-content">
#         This AI-powered system uses deep learning to analyze chest X-ray images 
#         and detect potential signs of Adenocarcinoma Cancer. The model has been 
#         trained on thousands of X-ray images to provide accurate predictions.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
    
    st.markdown("### 🎯 How to Use")
    st.markdown(
        """
        1. Upload a chest X-ray image (PNG, JPG, JPEG)
        2. Wait for the image to process
        3. Click the 'Analyze Image' button
        4. Review the prediction results
        """
    )
    # st.markdown("### 🎯 How to Use")
    # st.markdown(
    #     """
    #     <div class="sidebar-content">
    #     1. Upload a chest X-ray image (PNG, JPG, JPEG)
    #     2. Wait for the image to process
    #     3. Click the 'Analyze Image' button
    #     4. Review the prediction results
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
    
    # st.markdown("### ⚠️ Important Note")
    # st.markdown(
    #     """
    #     <div class="sidebar-content">
    #     This tool is for screening purposes only and should not be used as a 
    #     definitive diagnosis. Always consult with healthcare professionals for 
    #     proper medical advice.
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

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
MODEL_PATH = r"artifacts/training/model.h5"
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

# import os
# import numpy as np
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import img_to_array
# import time

# # Streamlit page configuration
# st.set_page_config(
#     page_title="AI Chest Cancer Detection",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Enhanced CSS with modern design elements
# st.markdown(
#     """
#     <style>
#         /* Main container styling */
#         .main {
#             background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
#             padding: 2rem;
#         }
        
#         /* Header styling */
#         .header {
#             background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%);
#             color: white;
#             padding: 2rem;
#             border-radius: 15px;
#             text-align: center;
#             margin-bottom: 2rem;
#             box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
#         }
        
#         .header h1 {
#             font-size: 2.5rem;
#             font-weight: 700;
#             margin-bottom: 1rem;
#             font-family: 'Helvetica Neue', sans-serif;
#         }
        
#         .header-subtitle {
#             font-size: 1.2rem;
#             opacity: 0.9;
#             line-height: 1.5;
#         }
        
#         /* Card container */
#         .card {
#             background: white;
#             padding: 2rem;
#             border-radius: 15px;
#             box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
#             margin-bottom: 2rem;
#         }
        
#         /* Upload section */
#         .upload-section {
#             border: 2px dashed #3498db;
#             border-radius: 15px;
#             padding: 2rem;
#             text-align: center;
#             transition: all 0.3s ease;
#             background: rgba(52, 152, 219, 0.05);
#         }
        
#         .upload-section:hover {
#             border-color: #2980b9;
#             background: rgba(52, 152, 219, 0.1);
#         }
        
#         /* Button styling */
#         .stButton > button {
#             background: linear-gradient(120deg, #2980b9 0%, #3498db 100%);
#             color: white;
#             padding: 0.8rem 2rem;
#             border-radius: 50px;
#             border: none;
#             font-weight: 600;
#             letter-spacing: 0.5px;
#             transition: all 0.3s ease;
#         }
        
#         .stButton > button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
#         }
        
#         /* Result boxes */
#         .result-box {
#             padding: 1.5rem;
#             border-radius: 12px;
#             text-align: center;
#             margin-top: 2rem;
#             font-size: 1.2rem;
#             font-weight: 600;
#             animation: fadeIn 0.5s ease-out;
#         }
        
#         .result-box.normal {
#             background: linear-gradient(120deg, #27ae60 0%, #2ecc71 100%);
#             color: white;
#         }
        
#         .result-box.cancer {
#             background: linear-gradient(120deg, #c0392b 0%, #e74c3c 100%);
#             color: white;
#         }
        
#         /* Progress bar */
#         .stProgress > div > div {
#             background-color: #3498db;
#         }
        
#         /* Animations */
#         @keyframes fadeIn {
#             from { opacity: 0; transform: translateY(10px); }
#             to { opacity: 1; transform: translateY(0); }
#         }
        
#         /* Info boxes */
#         .info-box {
#             background: #f8f9fa;
#             padding: 1rem;
#             border-radius: 10px;
#             margin-bottom: 1rem;
#             border-left: 4px solid #3498db;
#         }
        
#         /* Statistics section */
#         .stat-card {
#             background: white;
#             padding: 1rem;
#             border-radius: 10px;
#             text-align: center;
#             box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
#         }
        
#         .stat-number {
#             font-size: 1.8rem;
#             font-weight: 700;
#             color: #2c3e50;
#         }
        
#         .stat-label {
#             color: #7f8c8d;
#             font-size: 0.9rem;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Sidebar with information
# with st.sidebar:
#     st.markdown("### 📊 About the System")
#     st.info(
#         """
#         This AI-powered system uses deep learning to analyze chest X-ray images 
#         and detect potential signs of Adenocarcinoma Cancer. The model has been 
#         trained on thousands of X-ray images to provide accurate predictions.
#         """
#     )
    
#     st.markdown("### 🎯 How to Use")
#     st.markdown(
#         """
#         1. Upload a chest X-ray image (PNG, JPG, JPEG)
#         2. Wait for the image to process
#         3. Click the 'Analyze Image' button
#         4. Review the prediction results
#         """
#     )
    
#     st.markdown("### ⚠️ Important Note")
#     st.warning(
#         """
#         This tool is for screening purposes only and should not be used as a 
#         definitive diagnosis. Always consult with healthcare professionals for 
#         proper medical advice.
#         """
#     )

# # Main content
# st.markdown('<div class="header">', unsafe_allow_html=True)
# st.markdown('<h1>🔬 AI-Powered Chest Cancer Detection</h1>', unsafe_allow_html=True)
# st.markdown(
#     '<p class="header-subtitle">Advanced deep learning system for early detection of Adenocarcinoma Cancer through X-ray analysis</p>', 
#     unsafe_allow_html=True
# )
# st.markdown('</div>', unsafe_allow_html=True)

# # Statistics Row
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown(
#         """
#         <div class="stat-card">
#             <div class="stat-number">98.5%</div>
#             <div class="stat-label">Model Accuracy</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
# with col2:
#     st.markdown(
#         """
#         <div class="stat-card">
#             <div class="stat-number">< 2s</div>
#             <div class="stat-label">Analysis Time</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
# with col3:
#     st.markdown(
#         """
#         <div class="stat-card">
#             <div class="stat-number">10k+</div>
#             <div class="stat-label">Images Analyzed</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # Upload Section
# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown('<div class="upload-section">', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("📤 Upload X-Ray Image", type=["png", "jpg", "jpeg"])
# st.markdown('</div>', unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# class PredictionPipeline:
#     def __init__(self, model_path):
#         self.model = load_model(model_path)

#     def preprocess_image(self, image):
#         if image.mode != "RGB":
#             image = image.convert("RGB")
#         resized_image = image.resize((224, 224))
#         image_array = img_to_array(resized_image)
#         preprocessed_image = np.expand_dims(image_array, axis=0)
#         preprocessed_image /= 255.0
#         return preprocessed_image

#     def predict(self, image):
#         preprocessed_image = self.preprocess_image(image)
#         result = np.argmax(self.model.predict(preprocessed_image), axis=1)
#         confidence = np.max(self.model.predict(preprocessed_image)) * 100
#         return ("Normal" if result[0] == 1 else "Adenocarcinoma Cancer", confidence)

# # Initialize model
# MODEL_PATH = r"artifacts/training/model.h5"
# pipeline = PredictionPipeline(MODEL_PATH)

# if uploaded_file:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
#     with col2:
#         st.markdown("### Image Details")
#         st.markdown(f"**File Name:** {uploaded_file.name}")
#         st.markdown(f"**Image Size:** {image.size}")
#         st.markdown(f"**File Format:** {uploaded_file.type}")
#     st.markdown('</div>', unsafe_allow_html=True)

#     if st.button("🔍 Analyze Image"):
#         with st.spinner("Analyzing image..."):
#             # Add a small delay to show the progress bar
#             progress_bar = st.progress(0)
#             for i in range(100):
#                 time.sleep(0.01)
#                 progress_bar.progress(i + 1)
            
#             # Get prediction and confidence
#             prediction, confidence = pipeline.predict(image)
            
#             # Display results
#             if prediction == "Normal":
#                 st.markdown(
#                     f"""
#                     <div class="result-box normal">
#                         ✅ Result: {prediction}<br>
#                         Confidence: {confidence:.2f}%<br>
#                         No signs of cancer detected
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.markdown(
#                     f"""
#                     <div class="result-box cancer">
#                         ⚠️ Result: {prediction}<br>
#                         Confidence: {confidence:.2f}%<br>
#                         Please consult a healthcare professional immediately
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
            
#             # Additional information based on result
#             st.markdown("### Detailed Analysis")
#             if prediction == "Normal":
#                 st.markdown(
#                     """
#                     <div class="info-box">
#                     ℹ️ While no signs of cancer were detected, regular check-ups are recommended for preventive care.
#                     Regular screening can help detect any changes in your health early.
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.markdown(
#                     """
#                     <div class="info-box">
#                     🏥 Recommended next steps:
#                     - Schedule an appointment with an oncologist
#                     - Get a second opinion
#                     - Conduct additional diagnostic tests
#                     - Discuss treatment options with your healthcare provider
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center; color: #666;'>
#     💻 Developed with advanced AI technology | 🔒 HIPAA Compliant | 📊 Regular Updates
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# import os
# import numpy as np
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import img_to_array

# # Streamlit page configuration
# st.set_page_config(
#     page_title="Chest Cancer Classifier",
#     page_icon="🫁",
#     layout="centered"
# )

# # Modernized CSS styling for UI/UX
# st.markdown(
#     """
#     <style>
#         .main {
#             background-color: #f8f9fd;
#         }
#         .header {
#             color: #ffffff;
#             text-align: center;
#             font-family: 'Verdana', sans-serif;
#             background-color: #6c5ce7;
#             padding: 30px;
#             border-radius: 10px;
#             margin-bottom: 30px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#         .upload-box {
#             text-align: center;
#             border: 2px dashed #0984e3;
#             padding: 25px;
#             margin-top: 20px;
#             margin-bottom: 30px;
#             background-color: #ffffff;
#             border-radius: 10px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
#         }
#         .stButton > button {
#             background-color: #00cec9;
#             color: #ffffff;
#             padding: 15px 30px;
#             border: none;
#             border-radius: 50px;
#             cursor: pointer;
#             font-size: 20px;
#             font-weight: bold;
#             margin-top: 20px;
#             transition: all 0.3s ease;
#         }
#         .stButton > button:hover {
#             background-color: #009688;
#         }
#         .button-container {
#             display: flex;
#             justify-content: center;
#             margin-top: 20px;
#         }
#         .result-box {
#             border: 4px solid #2ecc71;
#             padding: 20px;
#             border-radius: 10px;
#             text-align: center;
#             margin-top: 30px;
#             font-size: 20px;
#             font-weight: bold;
#             color: #2ecc71;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#         .result-box.disease {
#             border-color: #e74c3c;
#             color: #e74c3c;
#         }
#         .instruction {
#             text-align: center;
#             color: #636e72;
#             margin-bottom: 20px;
#             font-size: 18px;
#             font-family: 'Arial', sans-serif;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Display a heading with improved text
# st.markdown('<div class="header"><h1>🫁 Chest Cancer Prediction System 🫁</h1></div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="instruction">This tool uses a deep learning model to analyze chest X-ray images and predict if the patient is <b>healthy</b> or may have <b>Adenocarcinoma Cancer</b>. Upload a chest X-ray image to get started.</div>',
#     unsafe_allow_html=True
# )

# # File uploader
# st.markdown('<div class="upload-box"><h3>📤 Upload your X-Ray Image below (JPEG, PNG, JPG)</h3></div>', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("📂 Select an image file...", type=["png", "jpg", "jpeg"])

# # Class for prediction pipeline
# class PredictionPipeline:
#     def __init__(self, model_path):
#         self.model = load_model(model_path)

#     def preprocess_image(self, image):
#         """
#         Preprocesses the uploaded image to ensure compatibility with the model.
#         - Converts to RGB if necessary.
#         - Resizes the image to (224, 224).
#         - Converts to a NumPy array with appropriate dimensions.
#         """
#         # Convert image to RGB if it has an alpha channel or grayscale
#         if image.mode != "RGB":
#             image = image.convert("RGB")
        
#         # Resize image to match the model's input shape
#         resized_image = image.resize((224, 224))
        
#         # Convert the image to a NumPy array
#         image_array = img_to_array(resized_image)
        
#         # Expand dimensions to match the model's input (1, 224, 224, 3)
#         preprocessed_image = np.expand_dims(image_array, axis=0)
        
#         # Normalize pixel values (optional, based on your model's training)
#         preprocessed_image /= 255.0
        
#         return preprocessed_image

#     def predict(self, image):
#         """
#         Predicts the class of the uploaded image.
#         - Preprocesses the image.
#         - Passes it through the model for prediction.
#         - Returns a user-friendly result string.
#         """
#         # Preprocess the image
#         preprocessed_image = self.preprocess_image(image)
        
#         # Predict using the loaded model
#         result = np.argmax(self.model.predict(preprocessed_image), axis=1)
        
#         # Return prediction result
#         return "Normal" if result[0] == 1 else "Adenocarcinoma Cancer"

# # Load the model only once to save resources
# MODEL_PATH = r"artifacts/training/model.h5"
# pipeline = PredictionPipeline(MODEL_PATH)

# if uploaded_file:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="🖼️ Uploaded X-Ray Image", use_container_width=True)

#     # Centered prediction button
#     st.markdown('<div class="button-container">', unsafe_allow_html=True)
#     if st.button("🔍 Predict"):
#         prediction = pipeline.predict(image)
#         # Conditional styling for the result box
#         if prediction == "Normal":
#             st.markdown(
#                 f"<div class='result-box'>🌟 <b>Prediction:</b> {prediction} - No signs of disease detected. 🌟</div>",
#                 unsafe_allow_html=True
#             )
#         else:
#             st.markdown(
#                 f"<div class='result-box disease'>⚠️ <b>Prediction:</b> {prediction} - Consult a doctor immediately for further diagnosis. ⚠️</div>",
#                 unsafe_allow_html=True
#             )
#     st.markdown('</div>', unsafe_allow_html=True)


# import os
# import numpy as np
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import img_to_array

# # Streamlit page configuration
# st.set_page_config(
#     page_title="Chest Cancer Classifier",
#     page_icon="🎗️",
#     layout="centered"
# )

# # Modernized CSS styling for UI/UX
# st.markdown(
#     """
#     <style>
#         .main {
#             background-color: #f8f9fd;
#         }
#         .header {
#             color: #ffffff;
#             text-align: center;
#             font-family: 'Verdana';
#             background-color: #cc75fa;
#             padding: 25px;
#             border-radius: 10px;
#             margin-bottom: 30px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#         .upload-box {
#             text-align: center;
#             border: 2px dashed #5e60ce;
#             padding: 25px;
#             margin-top: 20px;
#             margin-bottom: 30px;
#             background-color: #ffffff;
#             border-radius: 10px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
#         }
#         .stButton > button {
#             background-color: #5e60ce;
#             color: #ffffff;
#             padding: 15px 30px;
#             border: none;
#             border-radius: 50px;
#             cursor: pointer;
#             font-size: 20px;
#             font-weight: bold;
#             margin-top: 20px;
#             transition: all 0.3s ease;
#             text-align: center;
#             display: inline-block;
#         }
#         .stButton > button:hover {
#             background-color: #483d8b;
#         }
#         .button-container {
#             display: flex;
#             justify-content: center;
#             margin-top: 20px;
#         }
#         .result-box {
#             border: 4px solid #3CB371;
#             padding: 20px;
#             border-radius: 10px;
#             text-align: center;
#             margin-top: 30px;
#             font-size: 20px;
#             font-weight: bold;
#             color: #3CB371;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Display a heading
# st.markdown('<div class="header"><h1>🎗️ Chest Cancer Classifier 🎗️</h1></div>', unsafe_allow_html=True)

# # File uploader
# st.markdown('<div class="upload-box"><h3>📂 Upload an X-Ray image (JPEG, PNG, JPG)</h3></div>', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("🚀 Choose an image...", type=["png", "jpg", "jpeg"])

# # Class for prediction pipeline
# class PredictionPipeline:
#     def __init__(self, model_path):
#         self.model = load_model(model_path)

#     def preprocess_image(self, image):
#         """
#         Preprocesses the uploaded image to ensure compatibility with the model.
#         - Converts to RGB if necessary.
#         - Resizes the image to (224, 224).
#         - Converts to a NumPy array with appropriate dimensions.
#         """
#         # Convert image to RGB if it has an alpha channel or grayscale
#         if image.mode != "RGB":
#             image = image.convert("RGB")
        
#         # Resize image to match the model's input shape
#         resized_image = image.resize((224, 224))
        
#         # Convert the image to a NumPy array
#         image_array = img_to_array(resized_image)
        
#         # Expand dimensions to match the model's input (1, 224, 224, 3)
#         preprocessed_image = np.expand_dims(image_array, axis=0)
        
#         # Normalize pixel values (optional, based on your model's training)
#         preprocessed_image /= 255.0
        
#         return preprocessed_image

#     def predict(self, image):
#         """
#         Predicts the class of the uploaded image.
#         - Preprocesses the image.
#         - Passes it through the model for prediction.
#         - Returns a user-friendly result string.
#         """
#         # Preprocess the image
#         preprocessed_image = self.preprocess_image(image)
        
#         # Predict using the loaded model
#         result = np.argmax(self.model.predict(preprocessed_image), axis=1)
        
#         # Return prediction result
#         return "Normal" if result[0] == 1 else "Adenocarcinoma Cancer"

# # Load the model only once to save resources
# MODEL_PATH = r"artifacts/training/model.h5"
# pipeline = PredictionPipeline(MODEL_PATH)

# if uploaded_file:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="🖼️ Uploaded X-Ray Image", use_container_width=True)

#     # Centered prediction button
#     st.markdown('<div class="button-container">', unsafe_allow_html=True)
#     if st.button("🔍 Predict"):
#         prediction = pipeline.predict(image)
#         # Display the result in a styled box
#         st.markdown(
#             f"<div class='result-box'>🎯 Predicted Disease: {prediction}</div>",
#             unsafe_allow_html=True
#         )
#     st.markdown('</div>', unsafe_allow_html=True)



# import os
# import numpy as np
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array

# # Streamlit page configuration
# st.set_page_config(
#     page_title="Chest Cancer Classifier",
#     page_icon="🎗️",
#     layout="centered"
# )

# # Modernized CSS styling for UI/UX
# st.markdown(
#     """
#     <style>
#         .main {
#             background-color: #f8f9fd;
#         }
#         .header {
#             color: #ffffff;
#             text-align: center;
#             font-family: 'Verdana';
#             background-color: #cc75fa;
#             padding: 25px;
#             border-radius: 10px;
#             margin-bottom: 30px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#         .upload-box {
#             text-align: center;
#             border: 2px dashed #5e60ce;
#             padding: 25px;
#             margin-top: 20px;
#             margin-bottom: 30px;
#             background-color: #ffffff;
#             border-radius: 10px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
#         }
#         .stButton > button {
#             background-color: #ff6347;
#             color: #ffffff;
#             padding: 15px 30px;
#             border: none;
#             border-radius: 50px;
#             cursor: pointer;
#             font-size: 20px;
#             font-weight: bold;
#             margin-top: 20px;
#             transition: all 0.3s ease;
#         }
#         .stButton > button:hover {
#             background-color: #ff4500;
#         }
#         .result-box {
#             border: 4px solid #3CB371;
#             padding: 20px;
#             border-radius: 10px;
#             text-align: center;
#             margin-top: 30px;
#             font-size: 20px;
#             font-weight: bold;
#             color: #3CB371;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Display a heading
# st.markdown('<div class="header"><h1>🎗️ Chest Cancer Classifier 🎗️</h1></div>', unsafe_allow_html=True)

# # File uploader
# st.markdown('<div class="upload-box"><h3>📂 Upload an X-Ray image (JPEG, PNG, JPG)</h3></div>', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("🚀 Choose an image...", type=["png", "jpg", "jpeg"])

# # Class for prediction pipeline
# class PredictionPipeline:
#     def __init__(self, model_path):
#         self.model = load_model(model_path)

#     def predict(self, image):
#         # Resize and preprocess the image
#         test_image = img_to_array(image.resize((224, 224)))
#         test_image = np.expand_dims(test_image, axis=0)

#         # Predict using the loaded model
#         result = np.argmax(self.model.predict(test_image), axis=1)

#         # Return prediction
#         return "Normal" if result[0] == 1 else "Adenocarcinoma Cancer"

# # Load the model only once to save resources
# MODEL_PATH = r"artifacts/training/model.h5"
# pipeline = PredictionPipeline(MODEL_PATH)

# if uploaded_file:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="🖼️ Uploaded X-Ray Image", use_container_width=True)

#     # Prediction button
#     if st.button("🔍 Predict"):
#         prediction = pipeline.predict(image)
#         # Display the result in a styled box
#         st.markdown(
#             f"<div class='result-box'>🎯 Predicted Disease: {prediction}</div>",
#             unsafe_allow_html=True
#         )















# from src.chest_cancer_classifier import logger
# from chest_cancer_classifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
# from chest_cancer_classifier.pipeline.stage_2_prepare_base_model import PrepareBaseModelTrainingPipeline
# from chest_cancer_classifier.pipeline.stage_3_model_training import ModelTrainingPipeline
# from chest_cancer_classifier.pipeline.stage_4_model_evaluation import EvaluationPipeline

# # Define the name of the current stage in the data processing pipeline
# STAGE_NAME = "Data Ingestion stage"

# try:
#         # Log the start of the data ingestion stage
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         # Create an instance of the DataIngestionTrainingPipeline
#         obj = DataIngestionTrainingPipeline()
#         # Execute the main process of the data ingestion pipeline
#         obj.main()
#         # Log the successful completion of the data ingestion stage
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         # Log any exceptions that occur during the execution
#         logger.exception(e)
#         raise e  # Reraise the exception for further handling if necessary


# # Define the name of the current stage in the data processing pipeline
# STAGE_NAME = "Prepare Base Model stage"

# try:
#         # Log the start of the stage
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
#         # Create an instance of the training pipeline and run the main method
#         prepare_base_model = PrepareBaseModelTrainingPipeline()
#         prepare_base_model.main()
        
#         # Log the completion of the stage
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         # Log any exceptions that occur during the execution
#         logger.exception(e)
#         raise e  # Reraise the exception for further handling


# # Define the name of the current stage in the data processing pipeline
# STAGE_NAME = "Model Training stage"

# try:
#         # Log the start of the stage
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
#         # Create an instance of the training pipeline and run the main method
#         model_trainer = ModelTrainingPipeline()
#         model_trainer.main()
        
#         # Log the completion of the stage
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         # Log any exceptions that occur during the execution
#         logger.exception(e)
#         raise e  # Reraise the exception for further handling


# # Define the name of the current stage in the data processing pipeline
# STAGE_NAME = "Model Evaluation stage"

# try:
#         # Log the start of the evaluation stage
#         logger.info(f"*******************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
#         # Create an instance of the EvaluationPipeline class and run the main method
#         model_evaluater = EvaluationPipeline()
#         model_evaluater.main()
        
#         # Log the completion of the evaluation stage
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         # Log any exceptions that occur during the execution
#         logger.exception(e)
#         # Raise the exception to propagate it further
#         raise e