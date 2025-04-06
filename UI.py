import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Jai Shree Ram - Signature Verifier", layout="wide")

# Inject custom Bhagwa theme CSS
st.markdown("""
    <style>
        body {
            background-image: url('https://i.imgur.com/f6YzRNh.jpeg');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 10px;
        }
        h1, h2, h3, h4, h5, h6, p, label, div, .stTextInput > div > div > input {
            color: #FF6F00 !important; /* Bhagwa */
        }
        .prediction-box {
            background-color: white;
            color: #FF6F00;
            font-weight: bold;
            font-size: 22px;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        .forged {
            background-color: #B71C1C;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Upload', 'Settings'],
                           icons=['house', 'cloud-upload', 'gear'],
                           menu_icon="cast", default_index=1)

# Top Nav Menu
menu_selection = option_menu(None, ["Home", "Upload", "Settings"],
                             icons=['house', 'cloud-upload', 'gear'],
                             orientation="horizontal",
                             styles={
                                 "container": {"padding": "0!important", "background-color": "#FF6F00"},
                                 "icon": {"color": "white", "font-size": "20px"},
                                 "nav-link": {"font-size": "18px", "color": "white", "--hover-color": "#FFA040"},
                                 "nav-link-selected": {"background-color": "white", "color": "#FF6F00"},
                             })

# Load model
model = load_model('forge_real_signature_model.h5')

# Preprocess image
def preprocess_image(img):
    img = img.resize((512, 512))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload section
if menu_selection == "Upload":
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.title("🖋️ Signature Forgery Detector")
    st.write("Upload a signature image to check if it's **Genuine** or **Forged**.")

    uploaded_file = st.file_uploader("📂 Upload signature image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_file = Image.open(uploaded_file).convert('RGB')
        st.image(image_file, caption='🖼 Uploaded Signature', use_column_width=True)

        processed_image = preprocess_image(image_file)
        prediction = model.predict(processed_image)
        probability = prediction[0][0]

        st.markdown(f"### 🔍 Confidence Score: `{probability:.4f}`")

        label = "✅ Genuine" if probability > 0.5 else "❌ Forged"
        result_class = "" if probability > 0.5 else "forged"

        st.markdown(f'<div class="prediction-box {result_class}">{label}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Home section
if menu_selection == "Home":
    st.markdown("""
        <style>
            .hero-container {
                background-color: rgba(255, 111, 0, 0.1);
                padding: 3rem 2rem;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(255, 111, 0, 0.3);
                margin-top: 2rem;
            }
            .hero-title {
                font-size: 48px;
                font-weight: bold;
                color: #FF6F00;
                text-shadow: 1px 1px 2px #fff2e6;
            }
            .hero-subtitle {
                font-size: 22px;
                color: #FF6F00;
                margin-top: 1rem;
            }
            .hero-image {
                width: 50%;
                border-radius: 15px;
                box-shadow: 0 0 20px rgba(255, 111, 0, 0.7);
                margin-top: 2rem;
            }
        </style>
        
        <div class="hero-container">
            <div class="hero-title">🚩 Jai Shree Ram Signature Verifier</div>
            <div class="hero-subtitle">
                Blending Technology with Dharma.<br>
                A Divine AI Tool for Signature Verification. Powered by Bhagwan Ram’s blessings 🙏
            </div>
            <img class="hero-image" src="https://www.freepik.com/free-ai-image/businessman-holding-pen-signs-important-contract-document-generated-by-ai_42883652.htm#fromView=keyword&page=1&position=11&uuid=76a8b11c-1f73-49d0-84e7-c6dd1ebf6494&query=Signature" alt="Shri Ram Image">
        </div>
    """, unsafe_allow_html=True)


# Settings section
if menu_selection == "Settings":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.header("⚙️ Settings")
    st.write("Coming soon: model reloads, theme selection, and more.")
    st.markdown('</div>', unsafe_allow_html=True)
