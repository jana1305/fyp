import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the dataset
file_path = "bluebike_data (1).csv"

# Check if file exists
if not os.path.exists(file_path):
    st.error(f"❌ CSV file not found: {file_path}")
    st.stop()

# Read the dataset
df = pd.read_csv(file_path)

# Preprocess
num_cols = ['casual_riders_count', 'member_riders_count', 
            'casual_rider_duration', 'member_rider_duration', 
            'travel_time', 'Temp(c)', 'rel_humidity', 'wspd', 'pres']

# Drop unnamed columns if any
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[num_cols] = scaler.fit_transform(df[num_cols])

# Load the saved model
model_file = "gradient_boosting_model.pkl"
if not os.path.exists(model_file):
    st.error(f"❌ Model file not found: {model_file}")
    st.stop()

model = joblib.load(model_file)

# Streamlit UI
st.title("🚴‍♂️ Bluebike Usage Prediction")

st.markdown("### Enter Input Features")
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.slider("Relative Humidity (%)", 0, 100)
wind_speed = st.number_input("Wind Speed", min_value=0.0)
pressure = st.number_input("Atmospheric Pressure", min_value=900.0, max_value=1100.0)
is_weekend = st.selectbox("Is it Weekend?", [0, 1])
is_holiday = st.selectbox("Is it Holiday?", [0, 1])
travel_time = st.number_input("Average Travel Time (minutes)", min_value=0.0)

# Predict
if st.button("Predict"):
    input_data = np.array([[temp, humidity, wind_speed, pressure, 
                            is_weekend, is_holiday, travel_time]])
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Bike Count: {prediction[0]:.2f}")
