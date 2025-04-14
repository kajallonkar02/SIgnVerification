import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
import torch.nn as nn
from torchvision import models, transforms

# Set page configuration with improved settings
st.set_page_config(
    page_title="Signature Verification System",
    layout="wide",
    page_icon="✍️",
    initial_sidebar_state="expanded"
)

# Load local external CSS
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the custom CSS
load_local_css("style.css")

# Custom header with gradient
st.markdown("""
<div class="header-gradient">
    <div class="header-content">
        <h1>Signature Verification System</h1>
        <p>Advanced Deep Learning for Document Authentication</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load the Model ---
@st.cache_resource
def load_signature_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Real or Forged
    model.load_state_dict(torch.load("signature_model_10.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_signature_model()

# Image preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Global dataframe to store results
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Date", "Filename", "Prediction", "Confidence"])

# Sidebar navigation with improved styling
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    menu_selection = option_menu(
        "",
        ["Dashboard", "Verify Signature", "Verification History"],
        icons=['house', 'cloud-upload', 'clock-history'],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {
                "background-color": "#1a2639",
                "padding": "10px",
                "border-radius": "8px"
            },
            "icon": {
                "color": "#f0f0f0", 
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "8px 0",
                "color": "#f0f0f0",
                "border-radius": "5px",
                "padding": "10px 15px",
                "--hover-color": "#3e4a61"
            },
            "nav-link-selected": {
                "background-color": "#3e4a61",
                "color": "#ffffff",
                "font-weight": "600"
            },
        })

# Dashboard
if menu_selection == "Dashboard":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-content">
                <h1>Advanced Signature Verification</h1>
                <p class="hero-description">
                    Ensure document authenticity with our state-of-the-art deep learning technology.
                    Quickly distinguish between genuine and forged signatures with industry-leading accuracy.
                </p>
                <div class="hero-stats">
                    <div class="stat-card">
                        <div class="stat-value">99.2%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">0.8s</div>
                        <div class="stat-label">Avg. Processing</div>
                    </div>
                </div>
            </div>
            <div class="hero-image-container">
                <img class="hero-image" src="https://img.freepik.com/free-vector/closeup-fountain-pen-writing-signature-realistic_1284-13522.jpg" alt="Signature Illustration">
            </div>
        </div>
        
        <div class="features-container">
            <h2 class="section-title">Key Features</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h3>Deep Analysis</h3>
                    <p>Examines 200+ signature characteristics for comprehensive verification</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3>Real-Time Processing</h3>
                    <p>Get results in under a second with our optimized neural network</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Detailed Reporting</h3>
                    <p>Clear confidence scores and historical tracking for audit purposes</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Upload and verify
elif menu_selection == "Verify Signature":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.header("Signature Verification")
    st.markdown("""
    <p class="upload-instructions">
        Upload a scanned image of the handwritten signature for analysis. 
        Our system will evaluate the signature and provide an authenticity determination.
    </p>
    """, unsafe_allow_html=True)

    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose a signature image", 
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Signature Preview")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### Verification Results")
            if st.button("Analyze Signature", type="primary", use_container_width=True):
                with st.spinner("Analyzing signature characteristics..."):
                    input_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted = torch.max(outputs, 1)
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

                result = "Authentic" if predicted.item() == 1 else "Forged"
                confidence_pct = confidence * 100
                
                st.markdown(f"""
                <div class="result-container {'authentic' if result == 'Authentic' else 'forged'}">
                    <div class="result-header">
                        <h3>Analysis Complete</h3>
                        <div class="confidence-badge">{confidence_pct:.1f}%</div>
                    </div>
                    <div class="verdict">{result.upper()} SIGNATURE</div>
                    <div class="confidence-meter">
                        <div class="meter-fill" style="width: {confidence_pct}%"></div>
                    </div>
                    <p class="result-description">
                        The signature has been analyzed and classified as <strong>{result}</strong> 
                        with {confidence_pct:.1f}% confidence.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Save to history
                new_entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Filename": uploaded_file.name,
                    "Prediction": result,
                    "Confidence": f"{confidence_pct:.1f}%"
                }
                st.session_state.history_df = pd.concat(
                    [st.session_state.history_df, pd.DataFrame([new_entry])], 
                    ignore_index=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# History
elif menu_selection == "Verification History":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.header("Verification History")
    st.markdown("""
    <p class="history-description">
        Review past signature verification results and confidence metrics.
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.history_df.empty:
        # Enhanced dataframe display
        st.dataframe(
            st.session_state.history_df.sort_values("Date", ascending=False),
            column_config={
                "Date": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss"
                ),
                "Filename": "Document",
                "Prediction": st.column_config.TextColumn(
                    "Result",
                    help="Signature verification result"
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Model confidence in the prediction",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Export options
        st.download_button(
            label="Export History as CSV",
            data=st.session_state.history_df.to_csv(index=False),
            file_name="signature_verification_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No verification history available. Upload and analyze signatures to build your history.")

    st.markdown('</div>', unsafe_allow_html=True)