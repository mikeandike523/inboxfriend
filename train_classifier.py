#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neighbor-prefetch email classifier (MiniLM embeddings + soft kNN features + LogisticRegression).

What this script does (high level):
- Loads labeled messages from your DB (same Message/Classification models you used).
- Builds strict sender-agnostic text: "<subject>; <body>".
- Dedup + near-dedup the dataset.
- Group-aware split using normalized template keys (leakage-safe).
- Embeds all post-dedupe texts with MiniLM (L2-normalized), writes to a float16 np.memmap on disk.
- Trains a **LogisticRegression** head on **features derived from top-K (=500) neighbors** fetched from the
  training pool (leave-one-out for train points). Features are small, robust, and fast to compute.
- Evaluates on validation set (neighbors pulled from train pool only — like inference).
- Saves **everything needed for inference**: embeddings memmap path, index splits, labels, label maps,
  trained LR head, feature spec, and other metadata.

Notes:
- This replaces SetFit end-to-end training with a similarity-driven head while keeping your dedup/split logic.
- It intentionally keeps the pipeline RAM-light and scalable as N grows.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import joblib
from tqdm import tqdm

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

from sentence_transformers import SentenceTransformer

# --- Project-specific imports (match your environment) ---
from backend.models import Message, Classification
from backend.config import Config

# --------------------
# Defaults & constants
# --------------------
ENCODER_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_OUTDIR_DEFAULT = Path(__file__).parent / "backend" / "neighbor_prefetch_email_category"
NEAR_DUP_THRESH_DEFAULT = 0.94   # cosine sim threshold for near-duplicate removal
MINIMUM_SAMPLES_DEFAULT = 45     # merge categories smaller than this into "other"
TEST_SIZE_DEFAULT = 0.15
SEED_DEFAULT = 42
K_DEFAULT = 500                  # requested K
EMBED_D = 384                    # MiniLM-L6-v2 output dim

# --------------------
# Utils: text building
# --------------------

def build_text(subject: Optional[str], body: Optional[str]) -> str:
    s = (subject or "").strip()
    b = (body or "").strip()
    return f"{s}; {b}".strip()


def normalize_for_template(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|\bwww\.\S+', ' ', text)  # URLs
    text = re.sub(r'\d+', ' ', text)                        # numbers (dates/codes)
    text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&quot;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def assert_sender_agnostic(texts: List[str]) -> None:
    hits = sum(1 for t in texts if t.lower().startswith("from:"))
    if hits > 0:
        print(f"WARNING: {hits} texts seem to contain sender headers; inputs must be '<subject>; <body>'.")

# --------------------
# Data loading & cleaning
# --------------------

def merge_tiny_categories(labels: List[str], min_samples: int) -> List[str]:
    total = len(labels)
    counts = Counter(labels)
    to_merge = sorted([c for c, n in counts.items() if n < min_samples])
    if to_merge:
        print(f"\nMerging {len(to_merge)} tiny categories into 'other' (min {min_samples}):")
        for c in to_merge:
            n = counts[c]
            print(f"  - {c}: {n} samples ({100*n/total:.1f}%)")
    return ["other" if c in to_merge else c for c in labels]


def remove_exact_duplicates(texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
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


def near_dedup_embed(texts: List[str], encoder_name: str, out_path: Path) -> np.memmap:
    """Embed texts (float16 memmap, L2-normalized)."""
    st = SentenceTransformer(encoder_name)
    N = len(texts)
    mm = np.memmap(out_path, dtype=np.float16, mode='w+', shape=(N, EMBED_D))
    bs = 2048
    off = 0
    for i in tqdm(range(0, N, bs), desc="Embedding for near-dup"):
        batch = texts[i:i+bs]
        X = st.encode(batch, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
        mm[off:off+len(X)] = X.astype(np.float16)
        off += len(X)
    del st
    return mm


def remove_near_duplicates(texts: List[str], labels: List[str], thresh: float, encoder_name: str) -> Tuple[List[str], List[str]]:
    if len(texts) <= 1:
        return texts, labels
    tmp_emb = Path("near_dup_embs.f16.memmap")
    embs = near_dedup_embed(texts, encoder_name, tmp_emb)
    # brute-force cosine via dot on normalized vectors → distance = 1-cos
    X = np.asarray(embs, dtype=np.float32)
    keep = np.ones(len(texts), dtype=bool)
    for i in tqdm(range(len(texts)), desc="Near-dup scan"):
        if not keep[i]:
            continue
        q = X[i:i+1]  # [1, D]
        sims = (q @ X.T).ravel()  # cos
        sims[i] = -1.0
        dup_idx = np.where(sims >= thresh)[0]
        for j in dup_idx:
            if j > i:
                keep[j] = False
    kept_idx = np.where(keep)[0].tolist()
    X = None
    try:
        os.remove(tmp_emb)
    except Exception:
        pass
    return [texts[i] for i in kept_idx], [labels[i] for i in kept_idx]


def load_data(engine, min_samples: int, equivalency_map: Optional[Dict[str, Optional[str]]] = None):
    print("Loading training data from database…")
    with Session(engine) as s:
        categories = s.execute(select(Classification.category).distinct().order_by(Classification.category)).scalars().all()
        rows = s.execute(
            select(Message.subject, Message.content, Classification.category)
            .join(Classification, Classification.message_id == Message.id)
        ).all()

    print(f"Found classes: {categories}")
    print(f"Found {len(rows)} labeled messages")

    texts, labels_str = [], []
    for subject, content, category in tqdm(rows, desc="Processing messages"):
        text = build_text(subject, content)
        texts.append(text)
        labels_str.append(category)

    assert_sender_agnostic(texts)

    # Optional mapping phase (pre-merge)
    if equivalency_map:
        invalid_src = set(equivalency_map) - set(categories)
        if invalid_src:
            raise ValueError(f"Equivalency map has unknown source cats: {sorted(invalid_src)}")
        invalid_tgt = {t for t in equivalency_map.values() if t is not None and t != "other" and t not in categories}
        if invalid_tgt:
            raise ValueError(f"Equivalency map has invalid targets: {sorted(invalid_tgt)}")
        mapped = dropped = 0
        new_texts, new_labels = [], []
        for text, lbl in zip(texts, labels_str):
            if lbl in equivalency_map:
                tgt = equivalency_map[lbl]
                if tgt is None:
                    dropped += 1
                    continue
                new_lbl = "other" if tgt == "other" else tgt
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

    # Merge tiny categories
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

    # Dedup + near-dedup
    texts, labels_str = remove_exact_duplicates(texts, labels_str)
    texts, labels_str = remove_near_duplicates(texts, labels_str, thresh=NEAR_DUP_THRESH_DEFAULT, encoder_name=ENCODER_DEFAULT)

    # Label maps
    final_categories = sorted(set(labels_str))
    label_to_id = {c: i for i, c in enumerate(final_categories)}
    id2label = {i: c for c, i in label_to_id.items()}
    y = [label_to_id[c] for c in labels_str]

    # Groups for leakage-safe split
    groups = [normalize_for_template(t) for t in texts]

    return texts, y, final_categories, id2label, label_to_id, groups

# --------------------
# Split
# --------------------

def grouped_split(texts: List[str], y: List[int], groups: List[str], test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(texts, y, groups=groups))
    return train_idx.tolist(), val_idx.tolist()

# --------------------
# Embeddings to disk (for inference & training)
# --------------------

def embed_and_save(texts: List[str], encoder_name: str, out_path: Path) -> np.memmap:
    print("\nEmbedding all texts (MiniLM)…")
    st = SentenceTransformer(encoder_name)
    N = len(texts)
    mm = np.memmap(out_path, dtype=np.float16, mode='w+', shape=(N, EMBED_D))
    bs = 2048
    off = 0
    for i in tqdm(range(0, N, bs), desc="Embedding"):
        batch = texts[i:i+bs]
        X = st.encode(batch, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
        mm[off:off+len(X)] = X.astype(np.float16)
        off += len(X)
    del st
    mm.flush()
    print(f"Saved embeddings → {out_path} (float16 memmap)")
    return mm

# --------------------
# Neighbor prefetch (brute-force, batched; cosine via dot on unit vectors)
# --------------------

def topk_neighbors_for_queries(
    base_mm: np.memmap,
    q_indices: np.ndarray,
    pool_indices: np.ndarray,
    K: int,
    block: int = 50000,
    leave_one_out: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each query index in q_indices, fetch top-K neighbor indices & sims from pool_indices.
    Assumes both base and queries are unit-normalized (cosine == dot).
    Returns two lists (aligned with q_indices): [idxs_i], [sims_i].
    """
    X = base_mm  # [N, D] float16
    D = X.shape[1]
    Xf32 = None  # cast per block

    pool = pool_indices
    K_eff = min(K, max(1, len(pool) - (1 if leave_one_out else 0)))

    results_idx: List[np.ndarray] = []
    results_sim: List[np.ndarray] = []

    for qi in tqdm(q_indices, desc="Neighbor prefetch"):
        q = np.asarray(X[qi:qi+1], dtype=np.float32)  # [1, D]
        best_sim = np.full(K_eff, -2.0, dtype=np.float32)
        best_idx = np.full(K_eff, -1, dtype=np.int64)

        # Scan pool in blocks
        for s in range(0, len(pool), block):
            blk_idx = pool[s:s+block]
            B = np.asarray(X[blk_idx], dtype=np.float32)   # [B, D]
            sims = (q @ B.T).ravel()                       # [B]

            if leave_one_out:
                # If query appears in this block, invalidate self-sim.
                mask_self = (blk_idx == qi)
                if mask_self.any():
                    sims[mask_self.nonzero()[0][0]] = -2.0

            # Merge into running top-K (partial sort)
            if K_eff < len(sims):
                part_idx = np.argpartition(sims, -K_eff)[-K_eff:]
                cand_sims = sims[part_idx]
                cand_idx = blk_idx[part_idx]
            else:
                cand_sims = sims
                cand_idx = blk_idx

            # Combine candidates with current best and keep top K
            all_sims = np.concatenate([best_sim, cand_sims])
            all_idx = np.concatenate([best_idx, cand_idx])
            top = np.argpartition(all_sims, -K_eff)[-K_eff:]
            best_sim = all_sims[top]
            best_idx = all_idx[top]

        # sort descending
        order = np.argsort(-best_sim)
        results_idx.append(best_idx[order])
        results_sim.append(best_sim[order])

    return results_idx, results_sim

# --------------------
# Feature builder from neighbors
# --------------------

def neighbor_features(
    idx_list: List[np.ndarray],
    sim_list: List[np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    add_sorted_sims: bool = True,
) -> np.ndarray:
    """Build per-example features from its neighbors.
    Features: per-class sum/max/count (3*C), global max/mean/entropy/margin (4),
    optional sorted top-K sims (K).
    """
    feats: List[np.ndarray] = []
    ln = np.log

    for idxs, sims in zip(idx_list, sim_list):
        cls = labels[idxs]
        # per-class aggregates
        sum_c = np.zeros(num_classes, dtype=np.float32)
        max_c = np.full(num_classes, -1.0, dtype=np.float32)
        cnt_c = np.zeros(num_classes, dtype=np.float32)
        for c, s in zip(cls, sims):
            sum_c[c] += s
            cnt_c[c] += 1
            if s > max_c[c]:
                max_c[c] = s
        max_c[max_c < 0] = 0.0

        # global stats
        max_sim = float(np.max(sims)) if sims.size else 0.0
        mean_sim = float(np.mean(sims)) if sims.size else 0.0
        # entropy over class distribution
        if cnt_c.sum() > 0:
            p = cnt_c / max(1e-6, cnt_c.sum())
            ent = float(-(p[p>0] * np.log(p[p>0])).sum())
        else:
            ent = 0.0
        # margin: top1 - top2 over per-class sum
        top2 = np.sort(sum_c)[-2:]
        margin = float(top2[-1] - (top2[-2] if len(top2) > 1 else 0.0))

        vec = [*sum_c, *max_c, *cnt_c, max_sim, mean_sim, ent, margin]
        if add_sorted_sims:
            vec.extend(list(sims))
        feats.append(np.asarray(vec, dtype=np.float32))

    return np.vstack(feats)

# --------------------
# Threshold tuning (optional, like your original)
# --------------------

def tune_thresholds(probs: np.ndarray, y_true: np.ndarray, grid=(0.30, 0.80, 21)) -> np.ndarray:
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
    K: int,
    equivalency_map: Optional[Dict[str, Optional[str]]] = None,
) -> None:

    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + dedup
    texts, y, labels, id2label, label_to_id, groups = load_data(
        engine, min_samples=min_samples, equivalency_map=equivalency_map
    )

    N = len(texts)
    print(f"\nPost-dedupe examples: {N}")
    if N < 100:
        raise ValueError("Too few examples after dedupe (<100).")

    # Persist post-dedupe raw data for later reuse
    (model_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    with open(model_dir / "artifacts" / "texts.jsonl", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    np.save(model_dir / "artifacts" / "labels.npy", np.asarray(y, dtype=np.int32))
    with open(model_dir / "artifacts" / "id2label.json", "w") as f:
        json.dump({int(k): v for k, v in id2label.items()}, f, indent=2)
    with open(model_dir / "artifacts" / "label_to_id.json", "w") as f:
        json.dump(label_to_id, f, indent=2)

    # 2) Split (group-aware)
    train_idx, val_idx = grouped_split(texts, y, groups, test_size=test_size, seed=seed)
    y_arr = np.asarray(y, dtype=np.int32)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    if len(set(y_arr[train_idx])) < 2 or len(set(y_arr[val_idx])) < 2:
        raise ValueError("Training/validation split must each contain ≥2 classes.")

    # 3) Embed all texts (and save memmap for inference)
    emb_path = model_dir / "artifacts" / "embeddings.f16.memmap"
    E = embed_and_save(texts, encoder_name, emb_path)

    # 4) Neighbor K sanity
    # For TRAIN queries, pool is training set (leave-one-out); so maximum usable K is len(train)-1
    K_train_pool_max = max(1, len(train_idx) - 1)
    if K > K_train_pool_max:
        print(f"[warn] Requested K={K} > train-pool-1={K_train_pool_max}. Reducing K to {K_train_pool_max} for training.")
    K_train = min(K, K_train_pool_max)

    # For VAL queries, pool is training set; maximum usable K is len(train)
    K_val_pool_max = len(train_idx)
    K_val = min(K, K_val_pool_max)

    # 5) Neighbor prefetch → features
    train_idx_arr = np.asarray(train_idx, dtype=np.int64)
    val_idx_arr = np.asarray(val_idx, dtype=np.int64)

    # TRAIN features (LOO)
    tr_neighbors_idx, tr_neighbors_sim = topk_neighbors_for_queries(
        base_mm=E,
        q_indices=train_idx_arr,
        pool_indices=train_idx_arr,
        K=K_train,
        block=50000,
        leave_one_out=True,
    )
    Xtr = neighbor_features(tr_neighbors_idx, tr_neighbors_sim, labels=y_arr, num_classes=len(labels), add_sorted_sims=True)
    ytr = y_arr[train_idx_arr]

    # VAL features (pool=training, no LOO)
    va_neighbors_idx, va_neighbors_sim = topk_neighbors_for_queries(
        base_mm=E,
        q_indices=val_idx_arr,
        pool_indices=train_idx_arr,
        K=K_val,
        block=50000,
        leave_one_out=False,
    )
    Xva = neighbor_features(va_neighbors_idx, va_neighbors_sim, labels=y_arr, num_classes=len(labels), add_sorted_sims=True)
    yva = y_arr[val_idx_arr]

    # 6) Train LR head (with scaling). Keep small grid knobs via args if desired.
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", penalty="l2", multi_class="auto", n_jobs=-1)),
    ])

    print("\nTraining LogisticRegression head on neighbor features…")
    clf.fit(Xtr, ytr)

    # 7) Evaluate
    yva_pred = clf.predict(Xva)
    print("\nValidation report (LR on neighbor features):")
    print(classification_report([labels[i] for i in yva], [labels[i] for i in yva_pred], digits=3))

    # Optional: probability thresholds like your original
    if hasattr(clf.named_steps["lr"], "predict_proba"):
        probs = clf.predict_proba(Xva)
        th = tune_thresholds(probs, yva)
        meta_thresholds = th.tolist()
    else:
        meta_thresholds = None

    # 8) Persist artifacts
    joblib.dump(clf, model_dir / "neighbor_lr.joblib")

    meta = {
        "created_at": datetime.now().isoformat(),
        "encoder": encoder_name,
        "embedding_dim": EMBED_D,
        "embedding_memmap": str(emb_path.resolve()),
        "texts_jsonl": str((model_dir / "artifacts" / "texts.jsonl").resolve()),
        "labels_npy": str((model_dir / "artifacts" / "labels.npy").resolve()),
        "id2label_json": str((model_dir / "artifacts" / "id2label.json").resolve()),
        "label_to_id_json": str((model_dir / "artifacts" / "label_to_id.json").resolve()),
        "train_indices": list(map(int, train_idx)),
        "val_indices": list(map(int, val_idx)),
        "K_train": int(K_train),
        "K_val": int(K_val),
        "feature_spec": {
            "per_class": ["sum", "max", "count"],
            "globals": ["max_sim", "mean_sim", "entropy", "margin"],
            "sorted_sims": True
        },
        "thresholds": meta_thresholds,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved LR head → {model_dir / 'neighbor_lr.joblib'}")
    print(f"Saved metadata → {model_dir / 'metadata.json'}")
    print("Saved embeddings, labels, and text artifacts for inference.")

# --------------------
# CLI
# --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neighbor-prefetch email classifier")
    p.add_argument("--model-dir", type=Path, default=MODEL_OUTDIR_DEFAULT, help="Directory to save artifacts")
    p.add_argument("--test-size", type=float, default=TEST_SIZE_DEFAULT, help="Validation split proportion")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed")
    p.add_argument("--min-samples", type=int, default=MINIMUM_SAMPLES_DEFAULT, help="Merge categories with < this into 'other'")
    p.add_argument("--encoder", type=str, default=ENCODER_DEFAULT, help="SentenceTransformer backbone")
    p.add_argument("--equivalency-map", type=Path, default=None, help="JSON mapping old_category -> new|'other'|null (drop)")
    p.add_argument("--K", type=int, default=K_DEFAULT, help="Neighbor prefetch K (will be validated vs train size)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Prepare output dir
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Load optional map
    equivalency_map = None
    if args.equivalency_map and args.equivalency_map.exists():
        with open(args.equivalency_map, "r") as f:
            equivalency_map = json.load(f)

    # DB engine
    engine = create_engine(Config.DB_URL_EXTERNAL, future=True)

    train_model(
        engine=engine,
        model_dir=args.model_dir,
        test_size=args.test_size,
        seed=args.seed,
        min_samples=args.min_samples,
        encoder_name=args.encoder,
        K=args.K,
        equivalency_map=equivalency_map,
    )


if __name__ == "__main__":
    main()

# --------------------
# Inference pipeline (conceptual; implement in your service)
# --------------------
# 1) Load metadata.json, neighbor_lr.joblib, id2label.json, embeddings memmap, and labels.npy.
# 2) Build the TRAIN neighbor pool as the indices stored in metadata["train_indices"].
# 3) Given a new email, build text "<subject>; <body>", embed with the SAME encoder, L2-normalize.
# 4) Compute top-K neighbors against the TRAIN pool only (K=metadata["K_val"] is fine for inference).
# 5) Build the same feature vector via neighbor_features (use num_classes=len(id2label)).
# 6) Run the LR pipeline: probs = clf.predict_proba([features]) → apply optional per-class thresholds
#    stored in metadata["thresholds"] to pick the label; else argmax.
# 7) Optionally surface additional signals: max_sim, margin, entropy, or nearest exemplar subjects/bodies
#    (pull from texts.jsonl for explainability/debug UI).
