from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    loaded_model = joblib.load('mlp_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    loaded_model = None
    loaded_scaler = None


@app.route('/')
def home():
    return "🚀 EMG Prosthetic API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json(force=True)

        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid input: missing features'}), 400

        # Convert input to numpy array
        features = np.array(data['features'])

        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        scaled_features = loaded_scaler.transform(features)

        # Prediction
        prediction = loaded_model.predict(scaled_features)

        # Optional: Confidence (if model supports it)
        try:
            confidence = loaded_model.predict_proba(scaled_features)
            confidence = confidence.tolist()
        except:
            confidence = "Not available"

        return jsonify({
            'prediction': prediction.tolist(),
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ IMPORTANT: Required for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host='0.0.0.0', port=port)
