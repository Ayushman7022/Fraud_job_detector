# 🕵️‍♂️ Job Fraud Detection System

Welcome to the **Job Fraud Detection System** – a machine learning-powered solution to detect fraudulent job postings using smart analysis of job descriptions and salary patterns.

---

## 🚀 Overview

Job frauds are on the rise—fake job postings are wasting time and risking personal data. This system is designed to **analyze job descriptions and salary ranges** to predict whether a job is **genuine or fraudulent** using a **Logistic Regression model**.

### 🔍 What it does:
- Takes **job descriptions** and **salary info** as input
- Preprocesses the text and salary columns (removes `$`, cleans, vectorizes, etc.)
- Predicts whether the job is **fraudulent (1)** or **genuine (0)**
- Serves predictions via a **FastAPI backend**
- Provides an interactive interface using **Streamlit**

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Logistic Regression** (Scikit-learn)
- **Natural Language Processing (NLP)** (CountVectorizer)
- **FastAPI** 🔥 (Backend API)
- **Streamlit** 🌐 (Frontend interface)
- **Pickle** (Model serialization)

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
