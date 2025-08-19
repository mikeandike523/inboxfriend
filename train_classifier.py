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


def load_data(engine):
    with Session(engine) as s:
        rows = s.execute(
            select(
                Message.subject,
                Message.content,
                Message.sender_email,
                Classification.category,
            ).join(Classification)
        ).all()
    X = []
    y = []
    for subject, content, sender_email, category in rows:
        text = f"{subject or ''} {content or ''}".strip()
        domain = (
            sender_email.split("@")[-1]
            if sender_email and "@" in sender_email
            else ""
        )
        X.append((text, domain))
        y.append(category)
    return X, y


def train_model(engine, model_path: Path) -> None:
    X, y = load_data(engine)
    text_union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2))),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("text", text_union, 0),
            ("domain", OneHotEncoder(handle_unknown="ignore"), 1),
        ]
    )
    clf = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )
    clf.fit(X, y)
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
    engine = create_engine(Config.DB_URL, future=True)
    train_model(engine, args.model_path)


if __name__ == "__main__":
    main()
