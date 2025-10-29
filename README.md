# Vehicle-Heath-Assistant

The **AI Vehicle Health Assistant** is a machine learning-powered web application that analyzes a vehicle's real-time sensor data to determine its health status. The tool provides **predictive diagnostics** and alerts the user if the car may require maintenance — helping prevent breakdowns and improve vehicle longevity.

This app uses:
- **Python**
- **Scikit-Learn (Machine Learning Model)**
- **Streamlit (Web Interface)**


## Features

- Predicts vehicle condition based on sensor inputs
- Provides **specific issue explanations** (e.g., low oil, abnormal tire pressure, etc.)
- Simple and interactive **web-based UI**
- Runs locally with no external API required


## Inputs Used for Evaluation
| Sensor Data | Description |
|------------|-------------|
| Engine Temperature (°C) | Checks for overheating or cooling issues |
| Tire Pressure (psi) | Detects under/over inflation |
| Battery Voltage (V) | Identifies potential electrical issues |
| Oil Level (%) | Ensures engine lubrication is sufficient |
| Speed (mph) | Helps evaluate load and operation state |
| Mileage (mi) | Used to determine maintenance expectations |

## How To Clone Project:
git clone https://github.com/your-username/Vehicle-Health_Assistant.git
cd Vehicle-Health-Assistant

## Create and Activate the Environment:
python3 -m venv venv

- If using Mac or Linux use:
source venv/bin/activate     

- If using Windows use:
venv\Scripts\activate       

## Install Dependencies
pip install -r requirements.txt

## Ensure the Model Exists
python train_model.py

## Run the Web Application 
streamlit run app.py

The app will open in your brownwer automatically. However, if it doesnt work.. visit:
http://localhost:8501


```bash
python train_model.py
