# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load model
model = load_model('data/model.h5')

st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and the AI will recognize it!")

# Canvas for drawing
canvas = st.canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
)

if canvas.image_data is not None:
    img = Image.fromarray(canvas.image_data.astype('uint8')).convert('L')
    img = img.resize((28,28))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {digit}")
