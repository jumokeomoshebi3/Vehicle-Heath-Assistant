import streamlit as st
import pandas as pd 
import numpy as np 
import joblib 

#Loads the traioned model 
model = joblib.load ("car-health-model.pkl")

st.set_page_config(page_title="AI Vehicle Health Assistant", page_icon=":car:", layout="centered")
st.title("AI Vehicle Health Assistant")
st.write("Enter the vehicle sensor data to assess its health status.")

#user inputs
st.sidebar.header("Input Vehicle Sensor Data")

def get_user_input():
    engine_temp = st.sidebar.slider("Engine Temperature (°C)", 50, 150, 90)
    tire_pressure = st.sidebar.slider("Tire Pressure (psi)", 20, 40, 32)
    battery_voltage = st.sidebar.slider("Battery Voltage (V)", 10.0, 15.0, 12.5)
    oil_level = st.sidebar.slider("Oil Level (%)", 10, 100, 70)
    speed = st.sidebar.slider("Speed (km/h)", 0, 200, 60)

    user_data = {
        "engine_temp": engine_temp,
        "tire_pressure": tire_pressure,
        "battery_voltage": battery_voltage,
        "oil_level": oil_level,
        "speed": speed
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()
st.subheader("Vehicle Sensor Data Input")
st.dataframe(user_input)

#Make prediction
prediction = model.predict(user_input)[0]

#Display prediction
st.subheader("Predicted Vehicle Health Status")
st.success(prediction)

if prediction == "Normal :)":
    st.info("Your vehicle is in good health! Keep up the regular maintenance.")
elif prediction == "Battery Issue":
    st.warning("Warning: Your vehicle may have a battery issue. Consider checking the battery and charging system.")
elif prediction == "Overheating":
    st.error("Alert: Your vehicle is overheating! Stop driving and check the cooling system immediately.")
elif prediction == "Low Tire Pressure":
    st.warning("Warning: Your vehicle has low tire pressure. Check and inflate the tires to the recommended levels.")
elif prediction == "Low Oil":
    st.warning("Warning: Your vehicle has low oil levels. Check and refill the engine oil as needed.")
else:
    st.info("Status unknown. Please consult a professional mechanic for further diagnosis.")