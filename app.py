from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd # Needed if the input is expected as a structured format

app = Flask(__name__)
# run_with_ngrok(app) # Uncomment this line if you use flask-ngrok for public access

# Load the trained model and scaler
try:
    loaded_model = joblib.load('mlp_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully for API.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    loaded_model = None
    loaded_scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    data = request.get_json(force=True)

    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input: missing features'}), 400

    try:
        # Ensure input is a list of lists (for multiple samples) or a single list (for one sample)
        features = np.array(data['features'])
        if features.ndim == 1: # If a single sample is provided as a 1D array
            features = features.reshape(1, -1)

        # Scale the features
        scaled_features = loaded_scaler.transform(features)

        # Make prediction
        prediction = loaded_model.predict(scaled_features)

        # Convert prediction to a list to be JSON serializable
        # If the target was label encoded, you might want to inverse transform it here
        # For now, returning the numerical prediction
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# To run the app, you'd typically use `app.run()`
# In Colab, you might run it in a separate thread or use flask-ngrok.
# For simple testing within Colab, you can use the code below to run a local server.
# However, it will block the notebook cell.

# To demonstrate, I'll print instructions on how to test it once the cell is run.
print("Flask app is ready. To run it, execute `app.run(debug=False, port=5000)` in a *separate* cell or use `run_with_ngrok(app)` with `app.run()`.")
print("Example usage (after running the app):\n")
print("import requests\nurl = 'http://127.0.0.1:5000/predict' # or your ngrok URL\nheaders = {'Content-Type': 'application/json'}\n# Example with the first row of X_test\nexample_data = {'features': X_test.iloc[0].tolist()}\nresponse = requests.post(url, json=example_data, headers=headers)\nprint(response.json())")

# For directly running in Colab, this can block. A common pattern is to wrap it in a thread
# or use a tool like flask-ngrok which sets up a tunnel.
# DO NOT UNCOMMENT THIS DIRECTLY IF YOU WANT TO INTERACT WITH THE NOTEBOOK AFTERWARDS
# If you run this, you'll need to interrupt the kernel to stop the server.
# if __name__ == '__main__':
#     app.run(debug=False, port=5000)
