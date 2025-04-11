import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Job Fraud Detector")

# Input fields
description = st.text_area("Job Description")
min_salary = st.number_input("Min Salary", value=0.0)
max_salary = st.number_input("Max Salary", value=0.0)

if st.button("Predict"):
    # Vectorize
    desc_vector = vectorizer.transform([description])
    salary_features = np.array([[min_salary, max_salary]])
    full_features = np.hstack((desc_vector.toarray(), salary_features))

    pred = model.predict(full_features)[0]
    result = "🚨 Fraudulent" if pred == 1 else "✅ Legitimate"
    st.success(f"Prediction: {result}")
