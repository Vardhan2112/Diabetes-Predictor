# Diabetes Prediction Web Application

This is a web application for predicting whether a patient has diabetes based on key health metrics such as age, glucose level, insulin level, and BMI. The application uses machine learning models to make predictions and provides users with a simple, interactive interface for entering data and receiving predictions.

## Features

- **Multiple Models for Prediction**: Users can choose between different machine learning models to predict diabetes, including:
  - **Naive Bayes**
  - **Perceptron**
  - **Custom Perceptron** (a custom implementation of the Perceptron algorithm)
  
- **User Input Form**: Users can input personal data like age, glucose, insulin level, and BMI, and the app will predict if they have diabetes.

- **Real-time Predictions**: The app uses Flask as the backend to process the input data and return the prediction result.

## Demo

You can try the application locally by following the instructions below.

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, custom Perceptron algorithm
- **Data Handling**: Pandas
- **Model Serialization**: Pickle
- **Cross-Origin Resource Sharing (CORS)**: Enabled using Flask-CORS

## Installation

Follow these steps to get the project running locally on your machine.

### Prerequisites

Ensure you have the following installed:

- **Python 3.x**
- **pip** (Python package manager)

### Clone the Repository

1. First, clone this repository to your local machine:
```
git clone https://github.com/Vardhan2112/Diabetes-Predictor.git
```
2. Navigate to the project directory:
```
cd Diabetes-Predictor
```
### Set Up the Backend
1. Create a virtual environment (optional, but recommended):
```
python3 -m venv venv
```
2. Activate the virtual environment:

On Windows:
```
venv\Scripts\activate
```
On Mac/Linux:
```
source venv/bin/activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
### Running the Application
1. Start the Flask backend by running the following command:
```
python app.py
```
2. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
This will display the Diabetes Prediction app where you can input your data and get predictions.

### Frontend
The frontend is a simple HTML form where users can enter their health data (age, glucose, insulin, BMI) and choose a machine learning model for prediction. The form sends the input data to the Flask backend, which processes it and returns a prediction result.

### API Endpoints
/predict (POST)
Description: Accepts JSON data with user input for prediction and returns the model's prediction.

Input:
```
{
  "age": <float>,
  "glucose": <float>,
  "insulin": <float>,
  "bmi": <float>,
  "model": "<model_name>"
}
```
Where model_name can be one of:

"naive_bayes"
"perceptron"
"custom_perceptron"
Output:
```
{
  "prediction": 1,
  "message": "Model prediction says: Patient has diabetes"
}
or
{
  "prediction": 0,
  "message": "Model prediction says: Patient does not have diabetes"
}
```
### How the Model Works
The model is trained using a dataset that includes patient health data, such as glucose levels, insulin levels, BMI, and age. The application uses three different machine learning models to predict the likelihood of diabetes:

Naive Bayes: A probabilistic classifier based on Bayes' Theorem.\
Perceptron: A simple neural network model for binary classification.\
Custom Perceptron: A custom implementation of the Perceptron algorithm.
### Model Training and Deployment
The models are trained using scikit-learn, custom code.\
The trained models are serialized using Pickle for easy loading and reuse.\
The Flask backend loads the models and performs predictions based on user inputs.
### Contributions
Feel free to fork the repository and make improvements. Contributions are welcome!

To contribute:

Fork the repository.\
Create a new branch for your changes (git checkout -b feature-branch).\
Commit your changes (git commit -am 'Add feature').\
Push to your branch (git push origin feature-branch).\
Open a Pull Request.
