#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a SetFit email classifier with sender-agnostic inputs.

Key features:
- Input template: "<subject>; <body>"
- Dedup & near-dedup
- Group-aware split using normalized template keys
- TF-IDF + LogisticRegression backstop
- Confidence-gated blending + per-class thresholds (simulated post-calibration)
- Manual resampling (upsample minors, cap/don't upsample 'other')
- Configurable encoder; VRAM-aware batch sizes
"""

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
import joblib

# project-specific
from backend.models import Message, Classification
from backend.config import Config

# --------------------
# Defaults & constants
# --------------------

ACCELERATE_GB_DEFAULT = 8                         # soft GPU memory cap (GB)
MINIMUM_SAMPLES_DEFAULT = 45                      # merge categories smaller than this into "other"
NEAR_DUP_THRESH_DEFAULT = 0.94                    # cosine sim threshold to consider near-duplicates
CONF_GATE_TAU_DEFAULT = 0.60                      # SetFit confidence gate for lexical backstop
ENCODER_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_OUTDIR_DEFAULT = Path(__file__).parent / "backend" / "setfit_email_category"

# --------------------
# Utils: text building
# --------------------

def build_text(subject: Optional[str], body: Optional[str]) -> str:
    """Sender-agnostic, strict training/inference template."""
    s = (subject or "").strip()
    b = (body or "").strip()
    return f"{s}; {b}".strip()

def normalize_for_template(text: str) -> str:
    """Normalize text to derive template keys / leakage groups (no sender info)."""
    text = text.lower()
    text = re.sub(r'https?://\S+|\bwww\.\S+', ' ', text)  # URLs
    text = re.sub(r'\d+', ' ', text)                      # numbers (dates/codes)
    text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&quot;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def assert_sender_agnostic(texts: List[str]) -> None:
    """Warn if anything looks like a raw header snuck in."""
    hits = sum(1 for t in texts if t.lower().startswith("from:"))
    if hits > 0:
        print(f"WARNING: {hits} texts seem to contain sender headers; "
              f"inputs must be strictly '<subject>; <body>'.")

# --------------------
# GPU configuration
# --------------------

def setup_gpu_acceleration(accelerate_gb: int) -> Tuple[torch.device, float]:
    """Configure GPU settings and return (device, total_gpu_gb)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = props.total_memory / (1024**3)
        print(f"GPU detected: {torch.cuda.get_device_name(0)} | {gpu_memory_gb:.1f} GB")
        print(f"Using per-process memory fraction target based on {accelerate_gb} GB")
        if gpu_memory_gb > 0:
            memory_fraction = min(accelerate_gb / gpu_memory_gb, 0.9)
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            except Exception:
                pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        return device, gpu_memory_gb
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu"), 0.0

def pick_batch_sizes(device: torch.device, encoder_name: str, gpu_memory_gb: float) -> Tuple[int, int]:
    """Heuristic VRAM-aware batch sizes for (embeddings, head)."""
    if device.type != "cuda":
        return 32, 32
    enc = encoder_name.lower()
    if any(k in enc for k in ["large", "xl"]):
        bs = 16
    elif any(k in enc for k in ["base", "mpnet", "roberta", "gte", "e5"]):
        bs = 48 if gpu_memory_gb >= 12 else 32
    else:  # small models (MiniLM, tiny)
        bs = 64
    return bs, bs

# --------------------
# Data loading & cleaning
# --------------------

def merge_tiny_categories(labels: List[str], min_samples: int) -> List[str]:
    """Merge categories with < min_samples into 'other'."""
    total = len(labels)
    counts = Counter(labels)
    to_merge = sorted([c for c, n in counts.items() if n < min_samples])
    if to_merge:
        print(f"\nMerging {len(to_merge)} tiny categories into 'other' (min {min_samples}):")
        for c in to_merge:
            n = counts[c]
            print(f"  - {c}: {n} samples ({100*n/total:.1f}%)")
    out = ["other" if c in to_merge else c for c in labels]
    return out

def remove_exact_duplicates(texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Drop exact duplicates keyed by (normalized_text, label)."""
    norm = [normalize_for_template(t) for t in texts]
    seen = set()
    keep_idx = []
    for i, (k, y) in enumerate(zip(norm, labels)):
        key = (k, y)
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    return [texts[i] for i in keep_idx], [labels[i] for i in keep_idx]

def remove_near_duplicates(
    texts: List[str],
    labels: List[str],
    thresh: float,
    encoder_name: str
) -> Tuple[List[str], List[str]]:
    """Remove near-duplicates via cosine similarity on sentence embeddings."""
    if len(texts) <= 1:
        return texts, labels
    print(f"\nComputing embeddings for near-duplicate filtering (encoder={encoder_name})...")
    st = SentenceTransformer(encoder_name)
    embs = st.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    print(f"Finding near-duplicates (cosine >= {thresh:.2f})...")
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(embs)
    keep = np.ones(len(texts), dtype=bool)
    for i in range(len(texts)):
        if not keep[i]:
            continue
        dists, idxs = nn.kneighbors(embs[i:i+1], n_neighbors=10, return_distance=True)
        for d, j in zip(dists[0], idxs[0]):
            if j <= i:
                continue
            sim = 1 - d
            if sim >= thresh:
                keep[j] = False
    kept_idx = np.where(keep)[0].tolist()
    print(f"Kept {len(kept_idx)}/{len(texts)} after near-dup filtering.")
    return [texts[i] for i in kept_idx], [labels[i] for i in kept_idx]

def load_data(
    engine,
    min_samples: int,
    equivalency_map: Optional[Dict[str, Optional[str]]] = None
) -> Tuple[List[str], List[int], List[str], Dict[int, str], Dict[str, int], List[str]]:
    """Load labeled messages; build <subject>; <body> texts; merge tiny cats; return groups."""
    print("Loading training data from database...")
    with Session(engine) as s:
        # fetch known categories
        categories = s.execute(
            select(Classification.category).distinct().order_by(Classification.category)
        ).scalars().all()

        # fetch labeled rows
        rows = s.execute(
            select(Message.subject, Message.content, Classification.category)
            .join(Classification, Classification.message_id == Message.id)
        ).all()

    print(f"Found classes: {categories}")
    print(f"Found {len(rows)} labeled messages")

    # build texts & labels
    texts, labels_str = [], []
    for subject, content, category in tqdm(rows, desc="Processing messages"):
        text = build_text(subject, content)
        texts.append(text)
        labels_str.append(category)

    assert_sender_agnostic(texts)

    # validate and apply equivalency map (pre-merge)
    if equivalency_map:
        invalid_src = set(equivalency_map) - set(categories)
        if invalid_src:
            raise ValueError(f"Equivalency map has unknown source cats: {sorted(invalid_src)}")
        invalid_tgt = {
            tgt for tgt in equivalency_map.values()
            if tgt is not None and tgt != "other" and tgt not in categories
        }
        if invalid_tgt:
            raise ValueError(f"Equivalency map has invalid targets: {sorted(invalid_tgt)}")

        mapped = dropped = 0
        new_texts, new_labels = [], []
        for text, lbl in zip(texts, labels_str):
            if lbl in equivalency_map:
                target = equivalency_map[lbl]
                if target is None:
                    dropped += 1
                    continue
                new_lbl = "other" if target == "other" else target
                if new_lbl != lbl:
                    mapped += 1
                new_labels.append(new_lbl)
            else:
                new_labels.append(lbl)
            new_texts.append(text)
        texts, labels_str = new_texts, new_labels
        print(f"Applied equivalency map: mapped {mapped}, dropped {dropped}")

    if not labels_str:
        raise ValueError("No training rows remain after mapping. Check your data/equivalency map.")

    # merge tiny categories into "other"
    before = Counter(labels_str)
    print("\nBEFORE FILTERING (raw labels):")
    total = len(labels_str)
    for c, n in sorted(before.items()):
        print(f"  {c}: {n} ({100*n/total:.1f}%)")

    labels_str = merge_tiny_categories(labels_str, min_samples=min_samples)

    after = Counter(labels_str)
    print("\nAFTER MERGE (final labels):")
    for c, n in sorted(after.items()):
        print(f"  {c}: {n} ({100*n/total:.1f}%)")
    print(f"Total samples: {total} | Total categories: {len(after)}")

    # exact and near-duplicate filtering on subject+body
    texts, labels_str = remove_exact_duplicates(texts, labels_str)
    texts, labels_str = remove_near_duplicates(texts, labels_str, thresh=NEAR_DUP_THRESH_DEFAULT, encoder_name=ENCODER_DEFAULT)

    # label maps
    final_categories = sorted(set(labels_str))
    label_to_id = {c: i for i, c in enumerate(final_categories)}
    id2label = {i: c for c, i in label_to_id.items()}
    y = [label_to_id[c] for c in labels_str]

    # groups for leakage-safe split: based purely on normalized subject+body
    groups = [normalize_for_template(t) for t in texts]

    return texts, y, final_categories, id2label, label_to_id, groups

# --------------------
# Splitting & resampling
# --------------------

def grouped_split(
    texts: List[str], y: List[int], groups: List[str], test_size: float, seed: int
) -> Tuple[List[str], List[str], List[int], List[int], List[int], List[int]]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(texts, y, groups=groups))
    X_train = [texts[i] for i in train_idx]
    X_val   = [texts[i] for i in val_idx]
    y_train = [y[i] for i in train_idx]
    y_val   = [y[i] for i in val_idx]
    return X_train, X_val, y_train, y_val, train_idx, val_idx

def resample_train(
    X: List[str], y: List[int], id2label: Dict[int, str], max_target: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """Upsample minor classes (except 'other'); cap/downsample 'other'."""
    counts = Counter(y)
    # target size = max of non-'other' classes (or provided)
    non_other = [cls for cls in counts if id2label[cls] != "other"]
    if not non_other:
        return X, y
    target = max(counts[c] for c in non_other) if max_target is None else max_target

    by_class = defaultdict(list)
    for i, cls in enumerate(y):
        by_class[cls].append(i)

    new_idx = []
    for cls, idxs in by_class.items():
        label_name = id2label[cls]
        if label_name == "other":
            # downsample 'other' to target (or keep as-is if smaller)
            k = min(len(idxs), target)
            new_idx.extend(np.random.RandomState(17).choice(idxs, size=k, replace=False).tolist())
        else:
            # upsample minority classes to target
            if len(idxs) >= target:
                new_idx.extend(idxs[:target])
            else:
                reps = target - len(idxs)
                new_idx.extend(idxs + np.random.RandomState(17).choice(idxs, size=reps, replace=True).tolist())

    np.random.RandomState(17).shuffle(new_idx)
    X_res = [X[i] for i in new_idx]
    y_res = [y[i] for i in new_idx]

    print("\nResampling summary (per-class counts -> after):")
    after_counts = Counter(y_res)
    for cls in sorted(after_counts):
        print(f"  {id2label[cls]}: {counts[cls]} -> {after_counts[cls]}")
    return X_res, y_res

# --------------------
# Models & thresholds
# --------------------

def build_setfit(label_list: List[str], encoder_name: str) -> SetFitModel:
    return SetFitModel.from_pretrained(
        encoder_name,
        labels=label_list,
        use_differentiable_head=True,
        head_params={"out_features": len(label_list)},
    )

def train_tfidf_lr(X_train: List[str], y_train: List[int]) -> Tuple[TfidfVectorizer, LogisticRegression]:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, strip_accents="unicode")
    Xtr = tfidf.fit_transform(X_train)
    lr = LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs", multi_class="auto")
    lr.fit(Xtr, y_train)
    return tfidf, lr

def setfit_probabilities(model: SetFitModel, X: List[str]) -> np.ndarray:
    """Get class probabilities from SetFit model."""
    try:
        return model.predict_proba(X)
    except Exception:
        # Fallback: obtain logits via internal call if available, then softmax
        logits = model(X)  # may not work depending on SetFit version
        logits = np.asarray(logits)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

def tune_thresholds(probs: np.ndarray, y_true: np.ndarray, grid=(0.30, 0.80, 21)) -> np.ndarray:
    """Pick per-class thresholds maximizing binary F1 per class on a held-out split."""
    num_labels = probs.shape[1]
    thresholds = np.full(num_labels, 0.5, dtype=float)
    lo, hi, steps = grid
    for c in range(num_labels):
        best_f1, best_t = 0.0, 0.5
        pc = probs[:, c]
        y_true_c = (y_true == c)
        for t in np.linspace(lo, hi, steps):
            y_hat = pc >= t
            f1 = f1_score(y_true_c, y_hat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds

def predict_with_thresholds(probs: np.ndarray, th: np.ndarray) -> np.ndarray:
    preds = []
    for row in probs:
        mask = row >= th
        if mask.any():
            preds.append(int(np.argmax(row * mask)))
        else:
            preds.append(int(np.argmax(row)))
    return np.array(preds)

# --------------------
# Evaluation helpers
# --------------------

def distance_bucket_eval(
    tfidf: TfidfVectorizer, X_train: List[str], y_train: List[int],
    X_val: List[str], y_val: List[int], preds: List[int], id2label: Dict[int, str]
) -> None:
    """
    Bucket validation examples by lexical distance to training (TF-IDF centroid of all training),
    and print macro F1 per bucket. Helps visualize off-distribution behavior.
    """
    Xtr = tfidf.transform(X_train)
    Xva = tfidf.transform(X_val)

    # global centroid (normalize rows → mean → normalize)
    def normalize_rows(mat):
        row_norms = np.sqrt((mat.multiply(mat)).sum(axis=1)).A1 + 1e-12
        return mat.multiply(1.0 / row_norms[:, None])

    Xtr_n = normalize_rows(Xtr)
    Xva_n = normalize_rows(Xva)
    centroid = Xtr_n.mean(axis=0)

    # cosine sim to centroid
    sim = (Xva_n @ centroid.T).A1
    dist = 1 - sim

    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    y_val = np.array(y_val)
    preds = np.array(preds)

    print("\nDistance-bucket macro F1 (lower=farther from training centroid):")
    for lo, hi in buckets:
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() == 0:
            continue
        y_true_b = y_val[mask]
        y_pred_b = preds[mask]
        # per-class F1 then macro
        labels = sorted(set(y_true_b) | set(y_pred_b))
        # quick macro F1 using sklearn report
        rep = classification_report(y_true_b, y_pred_b, output_dict=True, zero_division=0)
        macro_f1 = rep.get("macro avg", {}).get("f1-score", 0.0)
        print(f"  [{lo:.1f}, {hi:.1f}): n={mask.sum():4d}  macroF1={macro_f1:.3f}")

# --------------------
# Persistence
# --------------------

def save_label_metadata(
    model_dir: Path,
    categories: List[str],
    id2label: Dict[int, str],
    label_to_id: Dict[str, int],
    encoder_name: str,
    thresholds: Optional[np.ndarray] = None,
    tau: Optional[float] = None
) -> Path:
    meta = {
        "categories": categories,
        "id2label": {int(k): v for k, v in id2label.items()},
        "label_to_id": label_to_id,
        "num_labels": len(categories),
        "encoder_name": encoder_name,
        "thresholds": thresholds.tolist() if thresholds is not None else None,
        "confidence_gate_tau": float(tau) if tau is not None else None,
        "created_at": datetime.now().isoformat()
    }
    p = model_dir / "label_metadata.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved label metadata → {p}")
    return p

# --------------------
# Training orchestration
# --------------------

def train_model(
    engine,
    model_dir: Path,
    test_size: float,
    seed: int,
    min_samples: int,
    encoder_name: str,
    accelerate_gb: int,
    near_dup_thresh: float,
    tau: float,
    equivalency_map: Optional[Dict[str, Optional[str]]] = None,
    do_distance_eval: bool = True,
) -> None:

    # fresh outdir
    if model_dir.exists():
        shutil.rmtree(model_dir)
    (model_dir).mkdir(parents=True, exist_ok=True)

    # GPU
    device, gpu_gb = setup_gpu_acceleration(accelerate_gb)

    # Data
    texts, y, labels, id2label, label_to_id, groups = load_data(
        engine, min_samples=min_samples, equivalency_map=equivalency_map
    )

    # Split (leakage-safe)
    X_train, X_val, y_train, y_val, _, _ = grouped_split(texts, y, groups, test_size, seed)

    # TF-IDF + LR baseline
    tfidf, lr = train_tfidf_lr(X_train, y_train)
    Xva_tfidf = tfidf.transform(X_val)

    # Resampling (avoid 'other' dominance)
    X_train_bal, y_train_bal = resample_train(X_train, y_train, id2label)

    # HF datasets
    train_ds = Dataset.from_dict({"text": X_train_bal, "label": y_train_bal})
    val_ds   = Dataset.from_dict({"text": X_val,        "label": y_val})

    # SetFit
    sf_model = build_setfit(labels, encoder_name=encoder_name)
    bs_embed, bs_head = pick_batch_sizes(device, encoder_name, gpu_gb)
    args = TrainingArguments(
        batch_size=(bs_embed, bs_head),
        num_epochs=(2, 16),            # brief encoder tuning, longer head training
        num_iterations=28,             # more contrastive pairs
        use_amp=(device.type == "cuda"),
        warmup_proportion=0.1,
        sampling_strategy="none",      # we already resampled; if invalid in your SetFit, remove this line
        end_to_end=False,              # classic SetFit: contrastive + linear head
        seed=seed,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
    )

    trainer = Trainer(
        model=sf_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        column_mapping={"text": "text", "label": "label"},
    )

    print(f"\nStarting SetFit training on {device}...")
    print(f"Embedding/Head batch sizes: {bs_embed}/{bs_head}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trainer.train()
    print("\nEvaluating SetFit on validation split...")
    sf_metrics = trainer.evaluate()
    print("SetFit eval metrics:", sf_metrics)

    # Predictions
    y_val_str = [id2label[i] for i in y_val]

    # SetFit-only report
    sf_pred_labels = trainer.model.predict(X_val)
    print("\nSetFit classification report:")
    print(classification_report(y_val_str, sf_pred_labels, digits=3))

    # Probs
    sf_probs = setfit_probabilities(trainer.model, X_val)
    lr_probs = lr.predict_proba(Xva_tfidf)

    # Confidence-gated blend
    sf_max = sf_probs.max(axis=1)
    blended_probs = np.where((sf_max[:, None] >= tau), sf_probs, lr_probs)

    # Per-class thresholds (simulated calibration on val)
    y_val_arr = np.array(y_val)
    tuned_thresholds = tune_thresholds(blended_probs, y_val_arr, grid=(0.30, 0.80, 21))
    y_pred_tuned = predict_with_thresholds(blended_probs, tuned_thresholds)

    print("\nBlended (gated) + per-class thresholds classification report:")
    print(classification_report([id2label[i] for i in y_val_arr],
                                [id2label[i] for i in y_pred_tuned], digits=3))

    # Optional: distance-bucket eval on the tuned preds
    if do_distance_eval:
        distance_bucket_eval(tfidf, X_train, y_train, X_val, y_val, y_pred_tuned.tolist(), id2label)

    # Persist artifacts
    trainer.model.save_pretrained(str(model_dir))
    joblib.dump(tfidf, model_dir / "tfidf.joblib")
    joblib.dump(lr,    model_dir / "tfidf_lr.joblib")
    save_label_metadata(
        model_dir,
        categories=labels,
        id2label=id2label,
        label_to_id=label_to_id,
        encoder_name=encoder_name,
        thresholds=tuned_thresholds,
        tau=tau
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU cache cleared")

    print(f"\nSaved SetFit model → {model_dir}")
    print("Saved TF-IDF & LR artifacts; saved label metadata with thresholds.")

# --------------------
# CLI
# --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SetFit email classifier (sender-agnostic).")
    p.add_argument("--model-dir", type=Path, default=MODEL_OUTDIR_DEFAULT,
                   help="Directory to save the trained artifacts.")
    p.add_argument("--test-size", type=float, default=0.15, help="Validation split proportion.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--min-samples", type=int, default=MINIMUM_SAMPLES_DEFAULT,
                   help="Merge categories with fewer than this many samples into 'other'.")
    p.add_argument("--encoder", type=str, default=ENCODER_DEFAULT,
                   help="SentenceTransformer backbone (e.g., sentence-transformers/all-mpnet-base-v2).")
    p.add_argument("--accelerate-gb", type=int, default=ACCELERATE_GB_DEFAULT,
                   help="Soft per-process VRAM cap (GB) for CUDA.")
    p.add_argument("--near-dup-thresh", type=float, default=NEAR_DUP_THRESH_DEFAULT,
                   help="Cosine similarity threshold for near-duplicate removal.")
    p.add_argument("--tau", type=float, default=CONF_GATE_TAU_DEFAULT,
                   help="Confidence gate for SetFit before deferring to TF-IDF/LR.")
    p.add_argument("--equivalency-map", type=Path, default=None,
                   help="JSON mapping old_category -> new_category|'other'|null (null = drop).")
    p.add_argument("--distance-eval", action="store_true",
                   help="Print macro-F1 by lexical distance buckets.")
    return p.parse_args()

def main() -> None:
    # optional clean of legacy dirs
    for d in ["checkpoints", "backend/setfit_email_category"]:
        if os.path.isdir(d):
            shutil.rmtree(d)

    args = parse_args()

    # load optional equivalency mapping
    equivalency_map = None
    if args.equivalency_map:
        with open(args.equivalency_map, "r") as f:
            equivalency_map = json.load(f)

    # DB engine
    engine = create_engine(Config.DB_URL_EXTERNAL, future=True)

    # Train
    train_model(
        engine=engine,
        model_dir=args.model_dir,
        test_size=args.test_size,
        seed=args.seed,
        min_samples=args.min_samples,
        encoder_name=args.encoder,
        accelerate_gb=args.accelerate_gb,
        near_dup_thresh=args.near_dup_thresh,
        tau=args.tau,
        equivalency_map=equivalency_map,
        do_distance_eval=args.distance_eval,
    )

if __name__ == "__main__":
    main()
