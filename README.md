# ✏️ Handwritten Digit Recognizer AI

A slightly advanced AI project that recognizes handwritten digits (0-9) using a neural network.  
Includes a **Streamlit web interface** for drawing digits in real-time.

---

## 🔹 Features
- Neural network trained on MNIST dataset.
- Real-time digit prediction via web app.
- Clean, professional code with modular structure.
- Optional improvements: deploy online, use CNN for higher accuracy.

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- Streamlit & streamlit-drawable-canvas
- NumPy & Pillow

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/yourusername/digit-recognizer.git
Install dependencies:

bash
Copier le code
pip install -r requirements.txt
Train the model (saves to data/model.h5):

bash
Copier le code
python train_model.py
Run the web app:

bash
Copier le code
streamlit run app.py
