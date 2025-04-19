# app.py
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Calories Burnt Prediction App")

# User input form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    duration = st.number_input("Exercise Duration (min)", min_value=1, max_value=300, value=30)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=100)
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

    submit = st.form_submit_button("Predict Calories Burnt")

if submit:
    gender_numeric = 0 if gender == "Male" else 1
    features = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    st.success(f"Estimated Calories Burnt: **{prediction:.2f}** kcal")
