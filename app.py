# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model('plant_disease_prediction_model.h5')

# Class labels (customize this)
class_names = ['Healthy', 'Powdery Mildew', 'Rust', 'Blight']

st.title("🌿 Plant Disease Prediction")
st.write("Upload an image of a leaf to predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    if st.button('Predict'):
        img = img.resize((224, 224))  # Adjust size as per model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f'Prediction: **{predicted_class}**')
