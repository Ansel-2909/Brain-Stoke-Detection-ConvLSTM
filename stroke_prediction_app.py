import pandas as pd
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

# Load preprocessing objects
with open('label_encoders.pkl', 'rb') as f:
    le_dict = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load the pre-trained model
model = load_model('stroke_prediction_model.keras')

# Streamlit UI for user input
st.title("Stroke Risk Prediction")

# Input fields
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", options=["Yes", "No"])
ever_married = st.selectbox("Ever Married", options=["Yes", "No"])
work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job"])
Residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", options=["Formerly smoked", "Never smoked", "Smokes", "Unknown"])

# Button for prediction
if st.button("Predict Stroke Risk"):
    # Prepare the input data for prediction
    patient_data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Encode the categorical features
    for col in le_dict:
        if col in patient_data:
            try:
                patient_data[col] = le_dict[col].transform([patient_data[col]])[0]
            except ValueError:
                # Assign fallback for unseen labels
                st.warning(f"Unseen label in {col}: {patient_data[col]} — assigning fallback.")
                patient_data[col] = le_dict[col].classes_[0]  # Using the first class as fallback

    # Create a DataFrame for scaling
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure the input data has the same columns and order as during training
    patient_df = patient_df[feature_names]

    # Scale the input data
    try:
        patient_scaled = scaler.transform(patient_df)
    except ValueError as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

    # Predict using the model
    prediction = model.predict(patient_scaled)
    risk_level = "High Risk" if prediction[0][0] > 0.5 else "Low Risk"  # Adjust based on model output

    # Display the result
    st.write(f"**Risk Assessment:** {risk_level}")
    st.write(f"**Probability of Stroke:** {prediction[0][0]:.2%}")

    


