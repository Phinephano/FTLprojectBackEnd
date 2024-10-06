from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model and scaler
with open('landslide_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the correct feature order
correct_feature_order = ["Aspect", "Curvature", "Earthquake", "Elevation", "Flow", 
                         "Lithology", "NDVI", "NDWI", "Plan", "Precipitation", 
                         "Profile", "Slope"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the frontend
        data = request.get_json()

        # Extract the input values in the correct feature order
        feature_values = [float(data[feature]) for feature in correct_feature_order]

        # Create a DataFrame from the input values
        input_data = pd.DataFrame([feature_values], columns=correct_feature_order)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_scaled)

        # Determine the result
        if prediction[0] == 1:
            result = "The region is risky"
            suggestions = [
                "Install early warning systems to detect potential landslides.",
                "Avoid construction on steep slopes or unstable ground.",
                "Monitor rainfall and weather forecasts regularly.",
                "Plan evacuation routes in case of emergency."
            ]
        else:
            result = "The region is not risky"
            suggestions = [
                "Continue monitoring the region for environmental changes.",
                "Maintain vegetation cover to reduce the risk of future landslides.",
                "Ensure drainage systems are working properly to prevent erosion.",
                "Encourage sustainable land use practices."
            ]

        # Return the result as JSON
        return jsonify({
            'prediction_text': result,
            'suggestions': suggestions
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
