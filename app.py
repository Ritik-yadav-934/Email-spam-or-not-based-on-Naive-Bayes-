from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from joblib import load
import re

app = Flask(__name__)
CORS(app)

# Load model & vectorizer
model = load("models/spam_nb_model.pkl")
vectorizer = load("models/tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"subject:", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Home (Frontend)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    cleaned_text = clean_text(text)
    vectorized = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]

    return jsonify({
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "spam_probability": round(float(probability), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
