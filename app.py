import numpy as np
import joblib
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the exported model and scaler
try:
    model = joblib.load("logistic_regression_model.joblib")
    scaler = joblib.load("scaler.joblib")
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json(force=True)
        
        # Expected structure: {"features": [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]}
        features = np.array(data["features"]).reshape(1, -1)
        
        # Standardize the features using the loaded scaler
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = int(model.predict(scaled_features)[0])
        probability = model.predict_proba(scaled_features)[0].tolist()
        
        # Return prediction results
        result = {
            "status": "success",
            "prediction": prediction,
            "label": "Diabetes Patient" if prediction == 1 else "Non-Diabetic Patient",
            "probability_non_diabetic": probability[0],
            "probability_diabetic": probability[1]
        }
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
