from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()

# Define the input schema
class JobRequest(BaseModel):
    job_description: str
    min_salary: float
    max_salary: float

@app.post("/predict")
def predict_job(request: JobRequest):
    # Vectorize the job description
    desc_vector = vectorizer.transform([request.job_description])

    # Combine with raw salary features
    salary_feats = np.array([[request.min_salary, request.max_salary]])
    full_features = np.hstack((desc_vector.toarray(), salary_feats))

    # Predict
    pred = model.predict(full_features)[0]
    result = "Fraudulent" if pred == 1 else "Legitimate"
    return {"prediction": result}
