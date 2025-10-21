import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#simulating sensor data for vehile monitoring
num_samples = 500
data = {
    "engine_temp": np.random.normal(90, 10, num_samples),
    "tire_pressure": np.random.normal(32, 3, num_samples),
    "battery_voltage": np.random.normal(12.5,0.8, num_samples),
    "oil_level": np.random.normal(70, 15, num_samples),
    "speed": np.random.normal(60, 20, num_samples),
}

df = pd.DataFrame(data)

# Create labels (basic rules for now)
def label_row(row):
    if row ["battery voltage"] < 11.5:
        return "Battery Issue"
    elif row ["engine_temp"] > 110:
        return "Overheating"
    elif row ["tire_pressure"] < 28:
        return "Low Tire Pressure"
    elif row ["oil_level"] < 40:
        return "Low Oil"
    else:
        return "Normal :)"
    
    df["status"] = df.apply(label_row, axis=1)

# Split data
X = df.drop("status", axis=1)
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trainig the model
model = DecisionTreeClassifier (max_depth = 4, random_state = 42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print ("Accuracy:", accuracy_score(y_test, y_pred))

#Save the model 
joblib.dump(model, "vehicle_health_model.pkl")
print("Model saved as vehicle_health_model.pkl")


