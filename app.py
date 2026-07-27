
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Diabetic Retinopathy Prediction",
    page_icon="🩺",
    layout="centered"
)


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()


st.title("🩺 Diabetic Retinopathy Prediction")
st.write(
    "Enter the patient's details below to predict the likelihood of "
    "diabetic retinopathy, based on a trained machine learning model."
)
st.divider()


with st.form("patient_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
    with col2:
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=150.0, value=80.0)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=50.0, max_value=400.0, value=100.0)

    submitted = st.form_submit_button("Predict")


if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ *Retinopathy Likely* — Predicted probability: {probability*100:.1f}%")
        st.write("Recommend the patient consult an ophthalmologist for further screening.")
    else:
        st.success(f"✅ *No Retinopathy Likely* — Predicted probability: {probability*100:.1f}%")
        st.write("Low risk based on current inputs. Regular monitoring is still advised.")

    st.progress(float(probability))
    st.caption(f"Model confidence for 'Retinopathy': {probability*100:.1f}%")


st.divider()
st.caption("Built as part of a Data Science group project | Model trained on 6,000 patient records")