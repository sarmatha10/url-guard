"""Flask API + static frontend.   Run:  python -m backend.app"""
import json
import os

import joblib
from flask import Flask, jsonify, request, send_from_directory

from backend.features import explain, extract, vectorize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=os.path.join(ROOT, "frontend"), static_url_path="")
MODEL_PATH = os.path.join(ROOT, "models", "model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
SAFE_BELOW, MALICIOUS_ABOVE = 0.35, 0.70       # tune these on validation data
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def label_for(score):
    return "Safe" if score < SAFE_BELOW else ("Suspicious" if score < MALICIOUS_ABOVE else "Malicious")


def analyze(url):
    score = float(model.predict_proba([vectorize(url)])[0][1])
    return {"url": url, "label": label_for(score), "risk_score": round(score * 100, 1),
            "indicators": explain(url), "features": extract(url)}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify(error="Model not trained. Run: python -m backend.train"), 503
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url or len(url) > 2000:
        return jsonify(error="Please provide a URL (max 2000 characters)."), 400
    return jsonify(analyze(url))


@app.get("/api/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify(error="No metrics yet. Train the model first."), 404
    with open(METRICS_PATH) as fh:
        return jsonify(json.load(fh))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
