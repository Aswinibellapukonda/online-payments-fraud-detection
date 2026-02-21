from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():
    try:
        step = float(request.form["step"])
        type = float(request.form["type"])
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])
        isFlaggedFraud = float(request.form["isFlaggedFraud"])

        input_data = np.array([[step, type, amount,
                                oldbalanceOrg, newbalanceOrig,
                                oldbalanceDest, newbalanceDest,
                                isFlaggedFraud]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "Fraudulent Transaction 🚨"
        else:
            result = "Legitimate Transaction ✅"

        return render_template("submit.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
