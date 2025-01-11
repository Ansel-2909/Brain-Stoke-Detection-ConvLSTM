import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .big-font {
        font-size: 24px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🏥 Stroke Prediction System")
st.markdown("""
    This application uses machine learning to predict stroke risk based on various health and demographic factors.
    Please input your information below for a risk assessment.
""")

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])
    occupation = st.selectbox("Occupation Type", 
                            ["Private Job", "Self Employed", "Government Job", "Children", "Unemployed"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])

with col2:
    st.subheader("Health Information")
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking = st.selectbox("Smoking Status", 
                         ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=400.0, value=100.0)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('stroke_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except:
        st.error("Error: Model files not found. Please ensure model files are properly saved.")
        return None, None

# Preprocess input data
def preprocess_input(data_dict):
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Create label encoder
    le = LabelEncoder()
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Marital Status', 'Occupation Type', 'Residence Type',
                       'Hypertension', 'Heart Disease', 'Smoking Status']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# Prediction function
def predict_stroke(features, model, scaler):
    # Preprocess features
    processed_features = preprocess_input(features)
    
    # Scale features
    scaled_features = scaler.transform(processed_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)[0][1]
    
    return prediction[0], prediction_proba

# Create prediction button
if st.button("Predict Stroke Risk", type="primary"):
    # Create features dictionary
    features = {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation Type': occupation,
        'Residence Type': residence,
        'Hypertension': hypertension,
        'Heart Disease': heart_disease,
        'Smoking Status': smoking,
        'BMI': bmi,
        'Average Glucose Level': glucose
    }
    
    # Load model and make prediction
    model, scaler = load_model()
    if model is not None and scaler is not None:
        prediction, probability = predict_stroke(features, model, scaler)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create three columns for results
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            # Create a gauge chart for risk probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Stroke Risk", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkred"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 33], 'color': 'lightgreen'},
                        {'range': [33, 66], 'color': 'yellow'},
                        {'range': [66, 100], 'color': 'salmon'}
                    ],
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig)

            # Display risk level and recommendations
            risk_level = "Low" if probability < 0.33 else "Medium" if probability < 0.66 else "High"
            st.markdown(f"""
                ### Risk Assessment
                - **Risk Level**: {risk_level}
                - **Probability**: {probability*100:.1f}%
                
                ### Recommendations:
                """)
            
            if risk_level == "Low":
                st.success("""
                    - Maintain your current healthy lifestyle
                    - Continue regular check-ups
                    - Stay physically active
                    - Maintain a balanced diet
                """)
            elif risk_level == "Medium":
                st.warning("""
                    - Schedule a check-up with your healthcare provider
                    - Monitor your blood pressure regularly
                    - Increase physical activity
                    - Consider lifestyle modifications
                """)
            else:
                st.error("""
                    - Seek immediate medical consultation
                    - Strict monitoring of vital signs
                    - Follow medical professional's advice
                    - Make necessary lifestyle changes
                """)

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
        This prediction system uses a machine learning model trained on healthcare data. The model takes into account:
        - Demographic factors (age, gender, occupation)
        - Medical history (hypertension, heart disease)
        - Lifestyle factors (smoking status, BMI)
        - Clinical measurements (glucose levels)
        
        The prediction is based on statistical patterns found in historical data and should not be considered as medical advice.
        Always consult with healthcare professionals for proper medical diagnosis and treatment.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Created for educational purposes. Not for medical use.</p>
    </div>
""", unsafe_allow_html=True)