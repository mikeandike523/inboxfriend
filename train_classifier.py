
import argparse
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from backend.models import Message, Classification
from backend.config import Config
from tqdm import tqdm
import torch
import os
import shutil
from collections import Counter
import json
import datetime

# --- Option B deps ---
# pip install setfit "sentence-transformers<3" datasets scikit-learn accelerate
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if os.path.isdir("checkpoints"):
    shutil.rmtree("checkpoints")
if os.path.isdir("backend/setfit_email_category"):
    shutil.rmtree("backend/setfit_email_category")

# GPU Configuration
ACCELERATE_GB = 8  # GPU memory limit in GB

# Category merging configuration
MINIMUM_SAMPLES = 45  # Merge categories with less than 60 samples into "other"


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


def merge_tiny_categories(labels_str, min_samples=MINIMUM_SAMPLES):
    """Merge categories with less than minimum samples into 'other'."""
    total_samples = len(labels_str)
    category_counts = Counter(labels_str)
    
    print(f"\nApplying filter: minimum {min_samples} samples (out of {total_samples} total)")
    
    # Identify categories to keep vs merge
    keep_categories = set()
    merge_categories = set()
    
    for category, count in category_counts.items():
        if count >= min_samples:
            keep_categories.add(category)
        else:
            merge_categories.add(category)
    
    if merge_categories:
        print(f"Categories to merge into 'other': {sorted(merge_categories)}")
        # Show details of what's being merged
        for category in sorted(merge_categories):
            count = category_counts[category]
            percentage = count / total_samples * 100
            print(f"  - {category}: {count} samples ({percentage:.1f}%)")
    else:
        print("No categories need merging")
    
    # Apply merging
    merged_labels = []
    merge_count = 0
    for label in labels_str:
        if label in merge_categories:
            merged_labels.append("other")
            merge_count += 1
        else:
            merged_labels.append(label)
    
    if merge_count > 0:
        print(f"Merged {merge_count} samples from {len(merge_categories)} categories into 'other'")
    
    return merged_labels


def load_data(engine):
    """Load labeled messages and build raw text dataset (subject + content)."""
    print("Loading training data from database...")
    with Session(engine) as s:
        # Fetch distinct classes from the database
        distinct_stmt = select(Classification.category).distinct().order_by(Classification.category)
        categories = s.execute(distinct_stmt).scalars().all()
        print(f"Found classes: {categories}")

        # Fetch all labeled messages
        rows = s.execute(
            select(
                Message.subject,
                Message.content,
                Classification.category,
            ).join(Classification, Classification.message_id == Message.id)
        ).all()

    print(f"Found {len(rows)} labeled messages")

    texts, labels_str = [], []
    for subject, content, category in tqdm(rows, desc="Processing messages"):
        text = f"Subject: {subject or ''}\nBody:\n{content or ''}".strip()
        texts.append(text)
        labels_str.append(category)

    if not labels_str:
        raise ValueError("No training rows found. Ensure the DB has labeled data.")

    print(f"Prepared {len(texts)} training samples")
    
    # Show BEFORE distribution
    print("\n" + "="*60)
    print("BEFORE FILTERING - Original category distribution:")
    print("="*60)
    original_dist = Counter(labels_str)
    total_samples = len(labels_str)
    for category, count in sorted(original_dist.items()):
        percentage = count / total_samples * 100
        print(f"  {category}: {count} samples ({percentage:.1f}%)")
    print(f"Total categories: {len(original_dist)}")
    print(f"Total samples: {total_samples}")
    
    # Merge tiny categories into "other"
    labels_str = merge_tiny_categories(labels_str)
    
    # Show AFTER distribution
    print("\n" + "="*60)
    print("AFTER FILTERING - Final category distribution:")
    print("="*60)
    final_dist = Counter(labels_str)
    for category, count in sorted(final_dist.items()):
        percentage = count / total_samples * 100
        print(f"  {category}: {count} samples ({percentage:.1f}%)")
    print(f"Total categories: {len(final_dist)}")
    print(f"Total samples: {total_samples}")
    print("="*60)

    # Build final label mappings
    final_categories = sorted(set(labels_str))
    label_to_id = {label: idx for idx, label in enumerate(final_categories)}
    id2label = {idx: label for label, idx in label_to_id.items()}

    y = [label_to_id[label] for label in labels_str]
    return texts, y, final_categories, id2label

def build_model(label_list):
    """Load a SetFit model (sentence-transformer + linear head)."""
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=label_list,
        use_differentiable_head=True,
        head_params={"out_features": len(label_list)},
    )
    return model

def save_label_metadata(model_dir: Path, final_categories, id2label, label_to_id):
    """Save label mappings for use in other parts of the application."""
    metadata = {
        "categories": final_categories,
        "id2label": id2label,
        "label_to_id": label_to_id,
        "num_labels": len(final_categories),
        "created_at": datetime.now().isoformat()
    }
    
    metadata_path = model_dir / "label_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved label metadata to {metadata_path}")
    return metadata_path

def train_model(engine, model_dir: Path, test_size: float = 0.15, seed: int = 42) -> None:
    # Setup GPU/CPU
    device = setup_gpu_acceleration()

    texts, y, label_list, id2label = load_data(engine)

    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Build HuggingFace Datasets
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})

    model = build_model(label_list)

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
    y_val_str = [id2label[i] for i in y_val]
    y_pred = trainer.model.predict(X_val)
    print(classification_report(y_val_str, y_pred, digits=3))

    # Save the SetFit model directory (encoder + classifier)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_dir))
    print(f"Saved SetFit model to {model_dir}")

    save_label_metadata(model_dir, label_list, id2label, label_to_id)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU cache cleared")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetFit email classifier")
    parser.add_argument(
        "--model-dir",
        default=Path(__file__).parent / "backend" / "setfit_email_category",
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
