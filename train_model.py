# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and merge
exercise = pd.read_csv('exercise.csv')
calories = pd.read_csv('calories.csv')
data = pd.merge(exercise, calories, on='User_ID')

# Preprocess
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
data.drop('User_ID', axis=1, inplace=True)
X = data.drop('Calories', axis=1)
y = data['Calories']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
