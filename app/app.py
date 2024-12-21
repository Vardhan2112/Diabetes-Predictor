from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

class CustomPerceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # Convert y from {0, 1} to {-1, 1} for compatibility with the Perceptron algorithm
        y_ = np.where(y <= 0, -1, 1)

        # Training loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Calculate linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                # Update weights and bias if prediction is incorrect
                if y_predicted != y_[idx]:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        # Calculate the linear output and make predictions
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


# Load saved models
with open("models/naive_bayes_model.pkl", "rb") as nb_file:
    naive_bayes_model = pickle.load(nb_file)

with open("models/perceptron_model.pkl", "rb") as perceptron_file:
    perceptron_model = pickle.load(perceptron_file)

with open("models/Cperceptron.pkl", "rb") as custom_perceptron_file:
    custom_perceptron_model = pickle.load(custom_perceptron_file)

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def home():
    """
    Home route to indicate that the Diabetes Prediction API is running.
    """
    return "Diabetes Prediction API is running. Use the /predict endpoint."

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "The requested resource was not found. Please check the URL and try again."}), 404


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data['age']
    glucose = data['glucose']
    insulin = data['insulin']
    bmi = data['bmi']
    model_type = data['model']

    features = pd.DataFrame([[glucose, insulin, bmi, age]], columns=['Glucose', 'Insulin', 'BMI', 'Age'])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Select the model based on the input
    if model_type == "naive_bayes":
        model = naive_bayes_model
    elif model_type == "perceptron":
        model = perceptron_model
    elif model_type == "custom_perceptron":
        model = custom_perceptron_model
    else:
        return jsonify({"error": "Invalid model type"}), 400

    # Make prediction using the scaled features
    prediction = model.predict(features_scaled)
    result = int(prediction[0])  # Convert the prediction to a Python integer for JSON serialization

    # Generate a user-friendly message
    message = "Patient has diabetes" if result == 1 else "Patient does not have diabetes"

    # Return the prediction and the message as JSON
    return jsonify({"prediction": result, "message": f"Model prediction says: {message}"})


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
