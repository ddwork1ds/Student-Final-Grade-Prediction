
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Score Predictor", page_icon="🎓")

@st.cache_resource
def load_model():
    paths = [Path("best_model.pkl"), Path("models/best_model.pkl")]
    for p in paths:
        if p.exists():
            return joblib.load(p), str(p)
    return None, None

model, model_path = load_model()

st.title("🎓 Student Score Predictor")
st.write("Predict final exam score (0-10 scale)")

# Input form
col1, col2 = st.columns(2)

with col1:
    sex_encoded = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 15, 22, 16)
    failures = st.slider("Past Failures", 0, 5, 0)

with col2:
    higher_encoded = st.selectbox("Wants Higher Education", ["No", "Yes"])
    absences = st.slider("Absences", 0, 60, 0)
    G_Avg = st.slider("Average Grade", 0.0, 10.0, 5.0, 0.1)

# Prediction
if st.button("Predict Score"):
    if not model:
        st.error("Model not found")
    else:
        # Prepare input
        input_data = {
            'sex_encoded': 0 if sex_encoded == "Female" else 1,
            'age': age,
            'failures': failures,
            'higher_encoded': 1 if higher_encoded == "Yes" else 0,
            'absences': absences,
            'G_Avg': G_Avg
        }
        
        # Predict
        try:
            prediction = model.predict(pd.DataFrame([input_data]))[0]
            score = max(0.0, min(10.0, float(prediction)))
            
            # Display result
            st.success(f"**Predicted Score: {score:.1f} / 10**")
            st.progress(score / 10.0)
            
        except Exception as e:
            st.error("Prediction failed")

# Model info in sidebar
if model_path:
    st.sidebar.success(f"Model: {model_path}")