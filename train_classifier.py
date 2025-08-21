import argparse
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from backend.models import Message, Classification
from backend.config import Config
from tqdm import tqdm
import re
import torch
import os

# --- Option B deps ---
# pip install setfit "sentence-transformers<3" datasets scikit-learn accelerate
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# GPU Configuration
ACCELERATE_GB = 8  # GPU memory limit in GB

# Target categories - anything not in this list will be classified as "OTHER"
TARGET_CATEGORIES = ["MARKETING", "NEWSLETTER"]

LABEL_MAP_CANONICAL = {"MARKETING": 0, "NEWSLETTER": 1, "OTHER": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP_CANONICAL.items()}

NORMALIZE_EQUIV = {
    # Marketing variants
    "MARKETING": "MARKETING", "MKT": "MARKETING", "MKTG": "MARKETING",
    "PROMO": "MARKETING", "AD": "MARKETING", "ADVERT": "MARKETING",
    # Newsletter variants
    "NEWSLETTER": "NEWSLETTER", "NL": "NEWSLETTER", "NEWSLETTERS": "NEWSLETTER"
}

def setup_gpu_acceleration():
    """Configure GPU settings and check availability."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {gpu_memory_gb:.1f} GB")
        print(f"Using GPU memory limit: {ACCELERATE_GB} GB")
        if gpu_memory_gb > 0:
            memory_fraction = min(ACCELERATE_GB / gpu_memory_gb, 0.9)
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            except Exception:
                pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        return device
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")

def normalize_label(raw):
    if raw is None:
        return "OTHER"
    lab = re.sub(r"\s+", "", str(raw).upper())
    return NORMALIZE_EQUIV.get(lab, lab if lab in TARGET_CATEGORIES else "OTHER")

def load_data(engine):
    """Load labeled messages and build raw text dataset (subject + content)."""
    print("Loading training data from database...")
    with Session(engine) as s:
        rows = s.execute(
            select(
                Message.subject,
                Message.content,
                Classification.category,
            ).join(Classification, Classification.message_id == Message.id)
        ).all()

    print(f"Found {len(rows)} labeled messages")

    texts, labels = [], []
    for subject, content, category in tqdm(rows, desc="Processing messages"):
        text = f"{subject or ''} {content or ''}".strip()
        label = normalize_label(category)
        if label not in TARGET_CATEGORIES:
            label = "OTHER"
        texts.append(text)
        labels.append(label)

    if not labels:
        raise ValueError("No training rows found. Ensure the DB has labeled data.")

    print(f"Prepared {len(texts)} training samples")
    dist = {c: labels.count(c) for c in set(labels)}
    print("Category distribution:", dist)

    # Convert labels to ints for dataset compatibility; we’ll map back for reports
    y = [LABEL_MAP_CANONICAL[l] for l in labels]
    return texts, y

def build_model():
    """Load a SetFit model (sentence-transformer + linear head)."""
    # Pass label STRINGS so predict() returns strings
    label_list = [ID2LABEL[i] for i in range(len(ID2LABEL))]  # ["MARKETING","NEWSLETTER","OTHER"]
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=label_list,
        use_differentiable_head=True,
        head_params={"out_features": len(label_list)},
    )
    return model

def train_model(engine, model_dir: Path, test_size: float = 0.15, seed: int = 42) -> None:
    # Setup GPU/CPU
    device = setup_gpu_acceleration()

    texts, y = load_data(engine)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Build HuggingFace Datasets
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})

    model = build_model()

    # Two-phase training knobs: (embeddings phase, head phase)
    bs_embed = 64 if device.type == "cuda" else 32
    bs_head = 64 if device.type == "cuda" else 32

    args = TrainingArguments(
        batch_size=(bs_embed, bs_head),      # (embeddings, head)
        num_epochs=(1, 12),                  # brief encoder tuning, longer head training
        num_iterations=20,                   # generate contrastive pairs
        use_amp=(device.type == "cuda"),
        warmup_proportion=0.1,
        sampling_strategy="oversampling",    # helpful for imbalance
        end_to_end=False,                    # classic SetFit: contrastive + linear head
        seed=seed,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        column_mapping={"text": "text", "label": "label"},
    )

    print(f"Starting SetFit training on {device}...")
    print(f"Embedding/Head batch sizes: {bs_embed}/{bs_head}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    trainer.train()
    print("Evaluating on validation split...")
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Predict returns label strings because we passed label strings at model init
    y_val_str = [ID2LABEL[i] for i in y_val]
    y_pred = trainer.model.predict(X_val)
    print(classification_report(y_val_str, y_pred, digits=3))

    # Save the SetFit model directory (encoder + classifier)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_dir))
    print(f"Saved SetFit model to {model_dir}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU cache cleared")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetFit email classifier (MARKETING/NEWSLETTER/OTHER)")
    parser.add_argument(
        "--model-dir",
        default=Path(__file__).parent / "backend" / "setfit_marketing_newsletter_other",
        type=Path,
        help="Directory to save the trained SetFit model",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Validation split proportion"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    args = parser.parse_args()

    engine = create_engine(Config.DB_URL_EXTERNAL, future=True)
    train_model(engine, args.model_dir, test_size=args.test_size, seed=args.seed)

if __name__ == "__main__":
    main()
