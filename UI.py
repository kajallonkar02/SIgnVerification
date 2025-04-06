import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# ===== Custom CSS Styling =====
st.set_page_config(page_title="Signature Verification", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: black;
            background-color: white;
        }
        .main {
            background-color: black;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #004080;
            text-align: center;
        }
        .stButton>button {
            background-color: #004080;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0059b3;
        }
        .prediction-box {
            background-color: bl;
            padding: 1rem;
            border-radius: 10px;
            font-weight: bold;
            font-size: 20px;
            text-align: center;
            margin-top: 1.5rem;
        }
        .genuine {
            background-color: #d0f0c0;
            color: #2e7d32;
        }
        .forged {
            background-color: #ffe6e6;
            color: #b00020;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Load the model =====
model = load_model('forge_real_signature_model.h5')

# ===== Image Preprocessing =====
def preprocess_image(img):
    img = img.resize((512, 512))                    
    img_array = image.img_to_array(img)             
    img_array = img_array / 255.0                   
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array

# ===== UI Layout =====
st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("🖋️ Signature Forgery Detection")
st.markdown("Upload a signature image below and the AI will tell you whether it is **Genuine** or **Forged**.")

uploaded_file = st.file_uploader("Upload a signature image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file).convert('RGB')
    st.image(image_file, caption='📷 Uploaded Image', use_column_width=True)

    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)

    probability = prediction[0][0]
    st.markdown(f"### 🔍 Prediction Confidence: `{probability:.4f}`")

    if probability > 0.5:
        label = "✅ Genuine"
        st.markdown(f'<div class="prediction-box genuine">{label}</div>', unsafe_allow_html=True)
    else:
        label = "❌ Forged"
        st.markdown(f'<div class="prediction-box forged">{label}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
