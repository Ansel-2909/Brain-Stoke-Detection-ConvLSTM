import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load the data
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Data Preprocessing
# Drop duplicates and null values
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Remove 'id' column if present
data = data.drop(columns=['id'], errors='ignore')

# Convert categorical variables
data["age"] = data["age"].astype("int")
data = data[data["gender"] != "Other"]
data["hypertension"] = data["hypertension"].astype(int)
data["heart_disease"] = data["heart_disease"].astype(int)
data["stroke"] = data["stroke"].astype(int)

# Store column names for later use
feature_names = [col for col in data.columns if col != 'stroke']

# Encode categorical variables
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le_dict = {col: LabelEncoder().fit(data[col]) for col in categorical_columns}
for col in categorical_columns:
    data[col] = le_dict[col].transform(data[col])

# Split features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Scale features
sc = StandardScaler()
X_scaled = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create Neural Network Model
def create_model():
    model = Sequential([
        Dense(32, input_dim=X_train.shape[1], activation="relu", kernel_regularizer=regularizers.l1(0.001)),
        Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.001)),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Train the model
model = create_model()
history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Function to make predictions
def predict_stroke_risk(patient_data):
    patient_df = pd.DataFrame([patient_data])
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in patient_df.columns:
            patient_df[col] = le_dict[col].transform(patient_df[col])

    # Scale the features
    patient_scaled = sc.transform(patient_df)
    
    # Make prediction
    probability = model.predict(patient_scaled)[0][0]
    print(f"Raw model output (probability): {probability:.4f}")
    
    prediction = "High Risk" if probability > 0.5 else "Low Risk"
    return {'prediction': prediction, 'probability': probability}

# Example usage with correct features
sample_patient = {
    'gender': 'Female',
    'age': 65,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 150.0,
    'bmi': 28.0,
    'smoking_status': 'formerly smoked'
}

# Make prediction for sample patient
try:
    result = predict_stroke_risk(sample_patient)
    print("\nSample Patient Prediction:")
    print(f"Risk Assessment: {result['prediction']}")
    print(f"Probability of Stroke: {result['probability']:.2%}")
except Exception as e:
    print(f"Error making prediction: {str(e)}")

# Save the model
model.save('stroke_prediction_model.keras')
print("\nModel saved as 'stroke_prediction_model.keras'")

