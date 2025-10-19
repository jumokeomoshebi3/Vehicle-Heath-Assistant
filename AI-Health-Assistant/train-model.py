import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

num_samples = 500
data = {
    "engine_temp": np.random.normal(90, 10, num_samples),
    "tire_pressure": np.random.normal(32, 3, num_samples),
    "battery_voltage": np.random.normal(12.5,0.8, num_samples),
    "oil_level": np.random.normal(70, 15, num_samples),
    "speed": np.random.normal(60, 20, num_samples),
}

df = pd.DataFrame(data)

