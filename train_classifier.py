import argparse
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from backend.models import Message, Classification
from backend.config import Config
from tqdm import tqdm
import re

# --- New imports for Option B (SetFit) ---
# pip install setfit "sentence-transformers<3" datasets scikit-learn
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Target categories - anything not in this list will be classified as "OTHER"
TARGET_CATEGORIES = ["MARKETING", "NEWSLETTER"]

LABEL_MAP_CANONICAL = {"MARKETING": 0, "NEWSLETTER": 1, "OTHER": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP_CANONICAL.items()}

NORMALIZE_EQUIV = {
    # Marketing variants
    "MARKETING": "MARKETING", "MKT": "MARKETING", "MKTG": "MARKETING",
    "PROMO": "MARKETING", "AD": "MARKETING", "ADVERT": "MARKETING",
    # Newsletter variants
    "NEWSLETTER": "NEWSLETTER", "NL": "NEWSLETTER",
}

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
        # Map to canonical set or OTHER
        if label not in TARGET_CATEGORIES:
            label = "OTHER"
        texts.append(text)
        labels.append(label)

    if not labels:
        raise ValueError("No training rows found. Ensure the DB has labeled data.")

    print(f"Prepared {len(texts)} training samples")
    dist = {c: labels.count(c) for c in set(labels)}
    print("Category distribution:", dist)

    # Convert labels to ints for SetFit
    y = [LABEL_MAP_CANONICAL[l] for l in labels]
    return texts, y


def build_model():
    """Load a SetFit model (sentence-transformer + linear head)."""
    # all-MiniLM-L6-v2 is a strong, fast baseline
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=list(ID2LABEL.keys()),
    )
    return model


def train_model(engine, model_dir: Path, test_size: float = 0.15, seed: int = 42) -> None:
    texts, y = load_data(engine)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Build HuggingFace Datasets
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})

    model = build_model()

    # Trainer config: increase num_iterations for more contrastive pairs; 
    # adjust epochs based on dataset size and overfitting signs
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        num_iterations=20,  # number of contrastive sampling rounds
        num_epochs=1,       # classifier head epochs per iteration
        batch_size=32,
        column_mapping={"text": "text", "label": "label"},
    )

    print("Starting SetFit training...")
    trainer.train()

    print("Evaluating on validation split...")
    # trainer.evaluate() returns accuracy/f1, but we also want a full report
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    y_pred = trainer.model.predict(X_val)
    print(classification_report([ID2LABEL[i] for i in y_val], [ID2LABEL[i] for i in y_pred], digits=3))

    # Save the SetFit model directory (contains encoder + classifier)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_dir))
    print(f"Saved SetFit model to {model_dir}")


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
