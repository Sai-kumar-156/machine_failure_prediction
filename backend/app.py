from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO

# Set the Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
model = joblib.load('model/model.pkl')

# Define the same feature list used during training
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
type_encoding = {'L': 0, 'M': 1, 'H': 2}

# Failure suggestions
failure_suggestions = {
    0: {
        "name": "No Failure",
        "causes": "N/A",
        "prevention": "N/A"
    },
    1: {
        "name": "Tool Wear Failure",
        "causes": "Excessive use of the tool, high rotational speed, high torque.",
        "prevention": "Regularly replace or maintain tools, optimize rotational speed and torque settings."
    },
    2: {
        "name": "Heat Dissipation Failure",
        "causes": "High air temperature, insufficient cooling.",
        "prevention": "Improve cooling systems, monitor and control air temperature."
    },
    3: {
        "name": "Power Failure",
        "causes": "Electrical issues, high power demand.",
        "prevention": "Ensure stable power supply, check electrical connections and components regularly."
    },
    4: {
        "name": "Overstrain Failure",
        "causes": "Excessive torque, high rotational speed, overloading.",
        "prevention": "Reduce load, optimize torque and speed settings, perform regular maintenance."
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming data is sent as JSON
    df = pd.DataFrame(data)
    
    # Ensure the input data matches the expected format
    df['Type'] = df['productType'].map(type_encoding)
    df.rename(columns={
        'airTemperature': 'Air temperature [K]',
        'processTemperature': 'Process temperature [K]',
        'rotationalSpeed': 'Rotational speed [rpm]',
        'torque': 'Torque [Nm]',
        'toolWear': 'Tool wear [min]'
    }, inplace=True)
    df = df[features]  # Reorder the columns to match the training data
    
    # Make predictions
    predictions = model.predict(df)
    
    # Prepare response with suggestions
    response = []
    for pred in predictions:
        failure_info = failure_suggestions.get(pred, {})
        response.append({
            "failure": failure_info.get("name"),
            "causes": failure_info.get("causes"),
            "prevention": failure_info.get("prevention")
        })
    
    return jsonify(response)

# @app.route('/visualize', methods=['POST'])
# def visualize():
#     data = request.json  # Assuming data is sent as JSON
#     df = pd.DataFrame(data)

#     # Ensure the input data matches the expected format
#     df['Type'] = df['productType'].map(type_encoding)
#     df.rename(columns={
#         'airTemperature': 'Air temperature [K]',
#         'processTemperature': 'Process temperature [K]',
#         'rotationalSpeed': 'Rotational speed [rpm]',
#         'torque': 'Torque [Nm]',
#         'toolWear': 'Tool wear [min]'
#     }, inplace=True)
#     df = df[features]  # Reorder the columns to match the training data

#     # Generate predictions
#     predictions = model.predict(df)
#     df['Predictions'] = predictions

#     # Convert numerical columns back to the original names for better visualization labels
#     df.rename(columns={
#         'Air temperature [K]': 'Air Temperature',
#         'Process temperature [K]': 'Process Temperature',
#         'Rotational speed [rpm]': 'Rotational Speed',
#         'Torque [Nm]': 'Torque',
#         'Tool wear [min]': 'Tool Wear'
#     }, inplace=True)

#     # Extract input values and ideal values
#     input_values = df.iloc[0][['Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool Wear']].values
#     ideal_values = [290, 310, 1500, 50, 20]  # Replace with actual ideal values
#     print("Input Values:", input_values)
#     print("Ideal Values:", ideal_values)
#     # Plot input vs ideal parameter values
#     fig, ax = plt.subplots(figsize=(10, 6))
#     labels = ['Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool Wear']
#     x = range(len(labels))
#     width = 0.35  # Width of the bars

#     ax.bar(x, input_values, width, label='Input Values', color='blue')
#     ax.bar([p + width for p in x], ideal_values, width, label='Ideal Values', color='orange')

#     ax.set_ylabel('Values')
#     ax.set_title('Input vs Ideal Parameter Values')
#     ax.set_xticks([p + width / 2 for p in x])
#     ax.set_xticklabels(labels)
#     ax.legend()

#     # Save the plot to a BytesIO object
#     img = BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plt.close()

#     return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
