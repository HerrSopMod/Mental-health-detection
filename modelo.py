import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Subida de modelo
uploaded_model = st.file_uploader("Sube tu modelo entrenado (.h5)", type=["h5"])
if uploaded_model is not None:
    model = load_model(uploaded_model)

# Mapeo de índices a categorías de enfermedades
diseases = ['Sin Enfermedad', 'Estrés', 'Ansiedad', 'Depresión', 'Neutro']

# Subida de imagen para análisis
uploaded_image = st.file_uploader("Sube una imagen para analizar", type=["png", "jpg", "jpeg"])
if uploaded_image is not None and uploaded_model is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (48, 48))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) / 255.0
    img_input = np.expand_dims(img_gray, axis=(0, -1))

    # Predicción
    prediction = model.predict(img_input)
    disease_index = np.argmax(prediction)
    confidence = prediction[0][disease_index]
    st.write(f"Resultado: {diseases[disease_index]} (Confianza: {confidence:.2f})")
