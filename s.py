import streamlit as st
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Job Fraud Detector", page_icon="🕵️‍♂️", layout="centered")

# Load model and vectorizer
model = joblib.load("fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Custom title with style
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🕵️‍♂️ Job Fraud Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Check whether a job description is legitimate or fraudulent.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input fields
description = st.text_area("📝 Job Description", placeholder="Paste the job details here...")
min_salary = st.number_input("💰 Minimum Salary", min_value=0.0, step=500.0)
max_salary = st.number_input("💸 Maximum Salary", min_value=0.0, step=500.0)

st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict"):
    with st.spinner("Analyzing the job posting..."):
        # Vectorize inputs
        desc_vector = vectorizer.transform([description])
        salary_features = np.array([[min_salary, max_salary]])
        full_features = np.hstack((desc_vector.toarray(), salary_features))

        # Prediction
        pred = model.predict(full_features)[0]
        result = "🚨 Fraudulent" if pred == 1 else "✅ Legitimate"

        # Show result with style
        if pred == 1:
            st.error(f"**Prediction: {result}**\n\nBe cautious! This job might be a scam.")
        else:
            st.success(f"**Prediction: {result}**\n\nThis job looks safe and genuine. 👍")
