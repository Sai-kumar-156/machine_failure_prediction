import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/MYDHILI/OneDrive/Desktop/machine_failure/predictive_maintainance.csv')


# Encode the 'Type' column
type_encoding = {'L': 0, 'M': 1, 'H': 2}
df['Type'] = df['Type'].map(type_encoding)

# Map failure types to target variable
failure_mapping = {
    'Tool Wear Failure': 1,
    'Heat Dissipation Failure': 2,
    'Power Failure': 3,
    'Overstrain Failure': 4
}
df['Target'] = df['Failure Type'].map(failure_mapping)
df['Target'] = df['Target'].fillna(0).astype(int)

# Define features and target variable
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]
y = df['Target']

# Print target value counts
print("Target value counts:\n", y.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(max_depth=25, n_estimators=40, random_state=42)
model.fit(X_train, y_train)

# Predict and print accuracy
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save the model to a .pkl file
joblib.dump(model, 'model.pkl')
