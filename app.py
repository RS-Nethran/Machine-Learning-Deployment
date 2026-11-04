
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "logistic_regression_model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

# Load model and scaler when the application starts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")

except Exception as e:
    raise RuntimeError(
        f"Failed to load model or scaler: {e}"
    ) from e


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Diabetes Prediction API is running."
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check that the request contains valid JSON
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Request must contain a valid JSON object."
            }), 400

        if "features" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'features' in request JSON."
            }), 400

        # Convert the input to a numeric NumPy array
        try:
            features = np.asarray(
                data["features"], dtype=np.float64
            )
        except (TypeError, ValueError):
            return jsonify({
                "status": "error",
                "message": "Features must contain numeric values."
            }), 400

        # Validate the eight expected input features
        if features.shape != (8,):
            return jsonify({
                "status": "error",
                "message": (
                    "Exactly 8 features are required: "
                    "Pregnancies, Glucose, BloodPressure, "
                    "SkinThickness, Insulin, BMI, "
                    "DiabetesPedigreeFunction, Age."
                )
            }), 400

        if not np.all(np.isfinite(features)):
            return jsonify({
                "status": "error",
                "message": "Features cannot contain NaN or infinity."
            }), 400

        # Reshape into one sample with eight features
        features = features.reshape(1, 8)

        # Apply the same scaler used during model training
        scaled_features = scaler.transform(features)

        # Generate prediction
        prediction = int(model.predict(scaled_features)[0])

        result = {
            "status": "success",
            "prediction": prediction,
            "label": (
                "Diabetes Patient"
                if prediction == 1
                else "Non-Diabetic Patient"
            )
        }

        # Return class probabilities if supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_features)[0]
            classes = list(model.classes_)

            if 0 in classes:
                result["probability_non_diabetic"] = float(
                    probabilities[classes.index(0)]
                )

            if 1 in classes:
                result["probability_diabetic"] = float(
                    probabilities[classes.index(1)]
                )

        return jsonify(result), 200

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
