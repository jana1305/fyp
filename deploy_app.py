import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("gradient_boosting_model.pkl")

# Streamlit App Title
st.title("🚲 Bike Sharing Demand Prediction App")

# Input fields
temp = st.number_input("Temperature (°C)", value=25.0)
humidity = st.slider("Relative Humidity (%)", 0, 100, 50)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
pressure = st.number_input("Atmospheric Pressure (hPa)", value=1010.0)
is_weekend = st.selectbox("Is it a Weekend?", [0, 1])
is_holiday = st.selectbox("Is it a Holiday?", [0, 1])
travel_time = st.number_input("Average Travel Time (min)", value=20.0)

# Prediction
if st.button("Predict Bike Count"):
    features = np.array([[temp, humidity, wind_speed, pressure, is_weekend, is_holiday, travel_time]])
    prediction = model.predict(features)
    st.success(f"🔮 Predicted Number of Riders: {prediction[0]:.2f}")
