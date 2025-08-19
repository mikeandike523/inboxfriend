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

# Target categories - anything not in this list will be classified as "other"
TARGET_CATEGORIES = ["MARKETING", "NEWSLETTER"]

def load_data(engine):
    """Load labeled messages and build text features only."""
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
    
    X, y = [], []
    for subject, content, category in tqdm(rows, desc="Processing messages"):
        text = f"{subject or ''} {content or ''}".strip()
        X.append(text)
        
        # Map categories to our target set or "OTHER"
        if category.upper() in TARGET_CATEGORIES:
            y.append(category.upper())
        else:
            y.append("OTHER")

    if not y:
        raise ValueError("No training rows found. Ensure the DB has labeled data.")

    print(f"Prepared {len(X)} training samples")
    print(
        f"""
Category distribution:
{dict(zip(*zip(*[(cat, y.count(cat)) for cat in set(y)])))}
""".strip()
        )
    return X, y

def build_pipeline():
    """Create the preprocessing + classifier pipeline focused on text content."""
    print("Building ML pipeline...")
    
    # Enhanced text feature extraction for better keyword and phrase detection
    text_features = FeatureUnion(
        [
            # Word-level features (1-3 grams for better phrase capture)
            ("word", TfidfVectorizer(
                ngram_range=(1, 3), 
                min_df=2, 
                max_df=0.95,  # Remove very common words
                stop_words='english'
            )),
            # Character-level features for catching marketing patterns
            ("char", TfidfVectorizer(
                analyzer="char", 
                ngram_range=(3, 6), 
                min_df=2,
                max_df=0.95
            )),
        ]
    )

    clf = Pipeline(
        [
            ("features", text_features),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,  # Increased for better convergence
                    solver="saga",
                    n_jobs=-1,
                    verbose=1,
                    C=1.0,  # Regularization strength
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