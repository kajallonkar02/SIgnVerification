import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set page configuration
st.set_page_config(
    page_title="Signature Verification System",
    layout="wide"
)

# Load local external CSS
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the custom CSS
load_local_css("style.css")

# Load the trained model
model = load_model('forge_real_signature_model.h5')

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((512, 512))
    img_array = image.img_to_array(img)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    img_data  = preprocess_input(img_array)
    return img_data

# Global dataframe to store results
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Date", "Filename", "Prediction Result", "Confidence Score"])

# Sidebar navigation
with st.sidebar:
    menu_selection = option_menu(
        "",
        ["Home", "Verify Signature", "Verification History"],
        icons=['speedometer2', 'cloud-upload', 'clock-history'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "rgb(177 158 138)"},
            "icon": {"color": "rgb(189 39 191)", "font-size": "35px"},
            "nav-link": {
                "font-size": "20px",
                "color": "#00BFFF",
                "--hover-color": "#333"
            },
            "nav-link-selected": {
                "background-color": "#111",
                "color": "#00BFFF"
            },
        })

# Dashboard
if menu_selection == "Home":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Signature Verification System</div>
            <div class="hero-subtitle">
                <div id="hero-text">
                Ensure the authenticity of handwritten signatures with advanced deep learning technology.<br>
                Delivering fast, reliable, and secure verification for modern financial and administrative workflows.
                </div>            
            </div>
            <img class="hero-image" height="350px" width="250px" src="https://img.freepik.com/free-vector/closeup-fountain-pen-writing-signature-realistic_1284-13522.jpg" alt="Signature Illustration">
        </div>
    """, unsafe_allow_html=True)

# Upload and verify
elif menu_selection == "Verify Signature":
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.title("Upload Signature for Verification")
    st.write("Please upload a scanned image of the handwritten signature you wish to verify. The system will analyze the image and determine if it is **authentic** or **forged**.")

    uploaded_file = st.file_uploader("Upload Signature Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_file = Image.open(uploaded_file).convert('RGB')
        st.image(image_file, caption='Uploaded Signature Preview', use_container_width=True)
        # img = image.load_img(r"C:\Users\HP\OneDrive\Desktop\sign_data\test\Real\049\01_049.png", target_size=(512,512))
        # st.image(img, caption='Uploaded Signature Preview', use_column_width=True)

        image_data = preprocess_image(image_file)
        prediction = model.predict(image_data)
        a = np.argmax(model.predict(image_data), axis = 1)
        probability = prediction[0][0]
        st.markdown(f"### Prediction Confidence: `{probability:.4f}`")

        label = "Authentic" if a ==1 else "Forged"
        label_display = "✔️ Authentic Signature" if label == "Authentic" else "Forged Signature Detected"
        result_class = "" if label == "Authentic" else "forged"

        st.markdown(f'<div class="prediction-box {result_class}">{label_display}</div>', unsafe_allow_html=True)

        # Save to session history  
        new_entry = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": uploaded_file.name,
            "Prediction Result": label,
            "Confidence Score": round(float(probability), 4)
        }
        st.session_state.history_df = pd.concat([st.session_state.history_df, pd.DataFrame([new_entry])], ignore_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

# History
elif menu_selection == "Verification History":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.header("Verification History")
    st.write("Below is a log of previously processed signature verifications for your reference.")

    df = st.session_state.history_df

    # Styled display
    st.dataframe(df.style.set_properties(**{
        'background-color': '#000000',
        'color': '#00BFFF',
        'border-color': '#00BFFF'
    }), height=300, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

