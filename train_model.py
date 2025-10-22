import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#simulating sensor data for vehile monitoring
np.random.seed(42)
num_samples = 500

data = {
    "engine_temp": np.random.normal(90, 10, num_samples),
    "tire_pressure": np.random.normal(32, 3, num_samples),
    "battery_voltage": np.random.normal(12.5,0.8, num_samples),
    "oil_level": np.random.normal(70, 15, num_samples),
    "speed": np.random.normal(60, 20, num_samples),
    "mileage": np.random.normal(60000, 15000, num_samples),
}

df = pd.DataFrame(data)

# Define health status (0= bad, 1 = good)
df["status"] = (
    (df["engine_temp"].between(80, 100)) &
    (df["tire_pressure"].between(30, 35)) &
    (df["battery_voltage"].between(12, 13)) &
    (df["oil_level"].between(25, 35)) &
    (df["mileage"] < 100000)
).astype(int)

# Split data
X = df.drop("status", axis=1)
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trainig the model
model = RandomForestClassifier (n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print (f"Model trained with accuracy: {acc:.2f}")

#Save the model 
joblib.dump(model, "car_health_model.pkl")
print("Model tained and saved as car_health_model.pkl")


