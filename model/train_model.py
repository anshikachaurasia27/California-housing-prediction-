import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Features & target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Feature engineering
X["rooms_per_household"] = X["AveRooms"] / X["HouseAge"]
X["population_per_household"] = X["Population"] / X["AveOccup"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f" Model trained successfully | R2 Score: {r2:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")