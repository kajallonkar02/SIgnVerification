# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load model
model = load_model('forge_real_signature_model.h5')

# Preprocessing function
def preprocess_image(img):
    img = img.resize((512, 512))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🖋️ Signature Forgery Detection")
st.write("Upload a signature image to verify if it's genuine or forged.")

uploaded_file = st.file_uploader("Choose a signature image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Signature", use_column_width=True)

    with st.spinner("Analyzing..."):
        processed = preprocess_image(img)
        prediction = model.predict(processed)
        label = "Genuine ✅" if prediction[0][0] > 0.5 else "Forged ❌"
        st.success(f"**Prediction:** {label}")
