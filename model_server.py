
"""
Lightweight Flask server to host SetFit model for external classification requests.
Run this on the host machine (outside Docker) to serve predictions over HTTP.
"""
import os
import torch
from flask import Flask, request, jsonify
from setfit import SetFitModel

app = Flask(__name__)

def setup_gpu_acceleration():
    """Configure GPU settings and check availability."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {gpu_memory_gb:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        return device
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")

# Setup GPU
device = setup_gpu_acceleration()

# Directory where the trained SetFit model is saved
MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "setfit_email_category")

# Load model at module level and move to GPU
app.logger.info(f"Loading SetFit model from {MODEL_DIR}")
model = SetFitModel.from_pretrained(MODEL_DIR)
model = model.to(device)


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
