from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Initialize FastAPI
app = FastAPI()

# Load model and scaler
model = load_model('model/best_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Define request body format
class FeaturesRequest(BaseModel):
    features: list

# Prediction endpoint
@app.post('/predict')
async def predict(request: FeaturesRequest):
    features = np.array(request.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    prediction = model.predict(features_scaled)
    result = {'prediction': int(prediction[0][0] > 0.5)}
    return result
