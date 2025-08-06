from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib
import uvicorn

# Load the model
model = joblib.load("chronic_disease_multilabel_model.pkl")

# Define possible disease labels used during training
disease_labels = [
    "Diabetes", "Heart Disease", "Lung Cancer", "Kidney Disease", "Liver Disease"
]

# FastAPI app
app = FastAPI(title="Multi-disease Prediction API", description="Predicts presence of multiple diseases based on health inputs", version="1.0")

# Input model
class PatientData(BaseModel):
    Age: int = Field(..., description="Age of the patient in years (e.g., 45)")
    Gender: int = Field(..., description="0 = Female, 1 = Male")
    BMI: float = Field(..., description="Body Mass Index (e.g., 27.5)")
    Smoking_Status: int = Field(..., description="0 = No, 1 = Yes")
    Alcohol_Intake: int = Field(..., description="0 = No, 1 = Yes")
    Physical_Activity: int = Field(..., description="0 = No, 1 = Yes")
    Blood_Pressure: float = Field(..., description="Systolic BP value (e.g., 120)")
    Cholesterol_Level: float = Field(..., description="Total cholesterol level (e.g., 200)")
    Blood_Sugar_Level: float = Field(..., description="Fasting blood sugar level (e.g., 95)")

@app.post("/predict")
def predict_diseases(data: PatientData):
    # Feature Engineering: Derive required columns
    mean_bp = data.Blood_Pressure  # Simplified
    lifestyle_score = (
        (1 - data.Smoking_Status) +
        (1 - data.Alcohol_Intake) +
        data.Physical_Activity
    )

    # Create ordered input list matching model training features
    input_vector = [
        data.Age,
        data.Gender,
        data.BMI,
        data.Blood_Sugar_Level,
        data.Cholesterol_Level,
        data.Smoking_Status,
        data.Alcohol_Intake,
        data.Physical_Activity,
        mean_bp,
        lifestyle_score
    ]

    # Reshape and predict
    input_array = np.array(input_vector).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)

    # Map prediction to disease names
    results = {}
    for i, disease in enumerate(disease_labels):
        results[disease] = {
            "Prediction": bool(prediction[i]),
            "Confidence": round(probabilities[i][0][1] * 100, 2) if isinstance(probabilities[i], np.ndarray) else "N/A"
        }

    return {"input": data.dict(), "predictions": results}
