import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Page Setup ---
st.title("🖋️ Signature Verification System")
st.write("Upload a signature image to verify if it's **Real** or **Forged**.")

# --- Load the Model ---
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Real or Forged
    model.load_state_dict(torch.load("signature_model_10.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Transform for Images ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Signature", use_container_width=True)

    # Prediction Button
    if st.button("Verify Signature"):
        input_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

        label = "Real" if predicted.item() == 1 else "Forged"
        st.markdown(f"### Prediction: **{label.upper()}**")
        st.write(f"Confidence Score: `{confidence:.4f}`")
