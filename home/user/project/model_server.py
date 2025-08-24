
"""
Lightweight Flask server to host SetFit model for external classification requests.
Run this on the host machine (outside Docker) to serve predictions over HTTP.
"""
import os

from flask import Flask, request, jsonify
from setfit import SetFitModel

app = Flask(__name__)

# Directory where the trained SetFit model is saved
MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "setfit_marketing_newsletter_other")

# Load model at module level
app.logger.info(f"Loading SetFit model from {MODEL_DIR}")
model = SetFitModel.from_pretrained(MODEL_DIR)


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON payload with a list of texts under 'texts', returns predictions and optional probabilities."""
    data = request.get_json(force=True)
    texts = data.get("texts")
    if not isinstance(texts, list):
        return jsonify({"error": "Expected 'texts' as a list of strings"}), 400

    # Predict labels
    preds = model.predict(texts)

    # Predict probabilities if supported
    try:
        probas = model.predict_proba(texts)
        # convert to pure Python floats for JSON serialization
        probabilities = [list(map(float, p)) for p in probas]
    except Exception:
        probabilities = None

    result = {"predictions": preds}
    if probabilities is not None:
        result["probabilities"] = probabilities
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)
