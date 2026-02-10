from flask import Flask, request, jsonify

import re
from joblib import load


app = Flask(__name__)

# Load trained model and vectorizer 
model = load("./models/Spam_nb_model.pkl")
vectorizer = load("./models/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"subject:", "", text)     # remove 'subject:'
    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"[^a-z\s]", " ", text)    # keep letters + spaces
    text = re.sub(r"\s+", " ", text)         # FIX: collapse spaces (NOT \S+)
    return text.strip()

# home routes
@app.route("/",methods=["GET"])
def home():
    return "Spam Detection API is running"

# Production routes
@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide text"}), 400
    
    text = data["text"]
    cleaned_text = clean_text(text)

    vectorizer_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorizer_text)[0]
    probability = model.predict_proba(vectorizer_text)[0][1]

    return jsonify({
        "predication ": "Spam" if prediction == 1 else "Not Spam",
        "Spam Probability": round(float(probability),4)
    })

# run the app 
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)