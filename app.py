from flask import Flask, request, jsonify,render_template

import re
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


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
@app.route("/")
def home():
    return render_template("index.html")

# Production routes
@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
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