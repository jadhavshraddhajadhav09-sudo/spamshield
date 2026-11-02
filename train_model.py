"""
train_model.py — Train a spam detection model on the SMS Spam Collection dataset.

Workflow:
    1. Load and clean the CSV data.
    2. Build a scikit-learn Pipeline (TF-IDF + Logistic Regression).
    3. Evaluate on a stratified 80/20 test split.
    4. Persist the pipeline to model/spam_model.joblib.
"""

import os
import re
import string
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.joblib")


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUMBER_RE = re.compile(r"\b\d+\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Normalise a single message string for modelling.

    Steps:
        - lowercase
        - replace URLs      → 'urltoken'
        - replace emails    → 'emailtoken'
        - replace numbers   → 'numbertoken'
        - strip punctuation
    """
    text = text.lower()
    text = URL_RE.sub("urltoken", text)
    text = EMAIL_RE.sub("emailtoken", text)
    text = NUMBER_RE.sub("numbertoken", text)
    text = text.translate(PUNCT_TABLE)
    return text.strip()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- 1. Load data -----
    if not os.path.isfile(DATA_PATH):
        print(
            "ERROR: Dataset not found at", DATA_PATH,
            "\n\nPlease download the SMS Spam Collection dataset from Kaggle:",
            "\n  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
            "\n\nSave the file as  data/spam.csv  inside the project folder.",
        )
        sys.exit(1)

    print("Loading dataset …")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Keep only the two relevant columns (v1 = label, v2 = message).
    # Some versions of the CSV have extra unnamed columns; drop them.
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})

    # ---- 2. Clean data -----
    df.dropna(subset=["label", "message"], inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Dataset: {len(df)} messages  ({df['label'].value_counts().to_dict()})")

    df["message"] = df["message"].apply(clean_text)

    # ---- 3. Train / test split -----
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        stratify=df["label"],
        random_state=42,
    )

    # ---- 4. Build pipeline -----
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )

    print("Training model …")
    pipeline.fit(X_train, y_train)

    # ---- 5. Evaluate -----
    y_pred = pipeline.predict(X_test)

    print("\n===== Evaluation on Test Set =====")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, pos_label='spam'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---- 6. Save model -----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
