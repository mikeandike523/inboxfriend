import argparse
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from backend.models import Message, Classification
from backend.config import Config
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from joblib import dump
from tqdm import tqdm

def load_data(engine):
    """Load labeled messages and build (text, domain) tuples."""
    print("Loading training data from database...")
    with Session(engine) as s:
        rows = s.execute(
            select(
                Message.subject,
                Message.content,
                Message.sender_email,
                Classification.category,
            ).join(Classification, Classification.message_id == Message.id)
        ).all()

    print(f"Found {len(rows)} labeled messages")
    
    X, y = [], []
    for subject, content, sender_email, category in tqdm(rows, desc="Processing messages"):
        text = f"{subject or ''} {content or ''}".strip()
        domain = (
            sender_email.split("@")[-1].lower().strip()
            if sender_email and "@" in sender_email
            else ""
        )
        X.append((text, domain))
        y.append(category)

    if not y:
        raise ValueError("No training rows found. Ensure the DB has labeled data.")

    print(f"Prepared {len(X)} training samples")
    return X, y

def build_pipeline():
    """Create the preprocessing + classifier pipeline."""
    print("Building ML pipeline...")
    
    text_union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("text", text_union, 0),
            ("domain", OneHotEncoder(handle_unknown="ignore"), [1]),
        ]
    )

    clf = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=250,
                    solver="saga",
                    n_jobs=-1,
                    verbose=1,  # Enable sklearn verbose output
                ),
            ),
        ]
    )
    return clf

def train_model(engine, model_path: Path) -> None:
    X, y = load_data(engine)
    clf = build_pipeline()
    
    print("Starting model training...")
    with tqdm(total=1, desc="Training model") as pbar:
        clf.fit(X, y)
        pbar.update(1)
    
    print(f"Saving model to {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, model_path)
    print(f"Saved model to {model_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train email classifier")
    parser.add_argument(
        "--model-path",
        default=Path(__file__).parent / "backend" / "classifier.joblib",
        type=Path,
        help="Where to save the trained model",
    )
    args = parser.parse_args()

    engine = create_engine(Config.DB_URL_EXTERNAL, future=True)
    train_model(engine, args.model_path)

if __name__ == "__main__":
    main()