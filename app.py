# app.py
"""
Streamlit Web App for Handwritten Digit Recognition
Users can draw digits and get predictions from trained AI model.
"""

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = load_model('data/model.h5')

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")
st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and the AI will predict it!")

# Create canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",        # background color
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img = ImageOps.invert(img)  # invert colors for white digit on black
    img = img.resize((28,28))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {digit}")
