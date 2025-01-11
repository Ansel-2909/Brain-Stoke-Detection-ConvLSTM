import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load your dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Replace 'N/A' and any other placeholders with NaN for proper handling
data.replace('N/A', pd.NA, inplace=True)

# Fill missing numerical values with the mean or any other strategy
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
data['avg_glucose_level'] = data['avg_glucose_level'].fillna(data['avg_glucose_level'].mean())

# Fill missing categorical values with 'Unknown'
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    data[col] = data[col].fillna('Unknown')

# Define the categorical columns to encode and numerical columns to scale
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

# Encode categorical columns
le_dict = {}
for col in categorical_columns:
    le = LabelEncoder()
    # Fit the encoder with all unique values and transform the data
    data[col] = le.fit_transform(data[col])  
    le_dict[col] = le  # Store the encoders for later use

# Save LabelEncoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

# Create the scaler for numerical columns
scaler = StandardScaler()
# Ensure consistent order by combining numerical and categorical columns
X_scaled = scaler.fit_transform(data[numerical_columns + categorical_columns])  

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the feature names used for scaling
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(numerical_columns + categorical_columns, f)

print("Label encoders, scaler, and feature names saved successfully!")
