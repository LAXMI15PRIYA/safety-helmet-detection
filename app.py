import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

# --- SIDEBAR ---
# Everything with "st.sidebar." appears on the left panel, not the main page.
st.sidebar.title("About This Project")
st.sidebar.write("This app detects whether workers are wearing safety helmets, using a YOLOv8 model fine-tuned on a construction site dataset.")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
st.sidebar.write("mAP50: 91.6%")
st.sidebar.write("Precision: 87%")
st.sidebar.write("Recall: 85%")

st.sidebar.markdown("---")
st.sidebar.subheader("How to Use")
st.sidebar.write("1. Upload an image using the uploader.\n2. Wait a moment for detection.\n3. View boxes and labels on the result.")

# --- MAIN PAGE ---
st.title("Safety Helmet Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(image)
    result_image = results[0].plot()
    st.image(result_image, caption="Detection Result", use_container_width=True)
