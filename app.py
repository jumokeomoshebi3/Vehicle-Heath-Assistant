import streamlit as st
#import pandas as pd 
import numpy as np 
import joblib 

#Loads the traioned model 
model = joblib.load ("car_health_model.pkl")

st.set_page_config(page_title="AI Vehicle Health Assistant", page_icon=":car:", layout="centered")
st.title("AI Vehicle Health Assistant")
st.write("Enter the vehicle sensor data to assess its health status.")

st.sidebar.header("About")
st.sidebar.info("This is an AI-powered Vehicle Health Assistant " \
"that predicts the health status of your vehicle based on sensor data " \
"inputs. Adjust the sliders to input your vehicle's sensor readings and get"\
" an instant health assessment!")

#user inputs
st.subheader("Please enter Vehicle Sensor Data:")


engine_temp = st.slider("Engine Temperature (°C)", 60, 120, 90)
tire_pressure = st.slider("Tire Pressure (psi)", 20, 40, 32)
battery_voltage = st.slider("Battery Voltage (V)", 10.0, 14.0, 12.5)
oil_level = st.slider("Oil Level (%)", 0, 100, 70)
speed = st.slider("Speed (MPH)", 0, 120, 60)
mileage = st.slider("Mileage (miles)", 0, 200000, 60000)

if st.button("Check Vehicle Health"):
    st.write("Running diagnostics...")
    issues = []
    prediction = None
    try:
        features = np.array([[engine_temp, tire_pressure, battery_voltage, oil_level, mileage]])
        prediction = model.predict(features)[0]
    except Exception as e:
        st.error(f"Error running diagnostics: {e}")

    if not (80 <= engine_temp <= 100):
        issues.append("Engine temperature is outside normal range - possible overheating or cooling issue. ")
    if not (30 <= tire_pressure <= 35):
        issues.append("Tire pressure is abnormal - check for leaks or improper inflation. ")
    if not (12 <= battery_voltage <= 13):
        issues.append("Battery voltage is abnormal - potential battery or alternator problem. ")
    if not (75 <= oil_level <= 100):
        issues.append("Oil level is not ideal - risk of engine damage. ")
    if mileage >= 100000:  
        issues.append("High mileage detected - consider more frequent maintenance. ")


    st.subheader("Diagnostic Results")
    if prediction == 1 and not issues:
        st.success("Your vehicle is in good health! :)")
        st.write("No issues detected based on the provided sensor data. Keep uo the good work!")
    elif prediction == 0 or issues: 
        st.error("Vehicle Status: ATTENTION NEEDED")
        if issues:
            st.warning("Detected Issues:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.info("The AI model detected an issue, but no specific sensor data flagged a problem. ")
    else:
        st.info("Status unknown - please consult a professional mechanic for a thorough checkup.")
    st.success("Diagnostics complete.")

st.markdown("---")
st.markdown("Developed by Jumoke Omoshebi 🚗💡")