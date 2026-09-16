"""TF-IDF + logistic regression baseline.

Worth having before any transformer: it trains in seconds, sets the bar the
fine-tune has to clear, and its failures point at which intents are genuinely
ambiguous rather than just hard for a small model.
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from data import SEED, load_splits

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "baseline-v1"


def main():
    (train_x, train_y), (val_x, val_y), _, labels = load_splits()

    pipeline = make_pipeline(
        # Char n-grams carry most of the signal here: support tickets are short
        # and full of near-identical phrasings that differ by a word or two.
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10.0, max_iter=2000, random_state=SEED),
    )
    pipeline.fit(train_x, train_y)

    val_macro_f1 = f1_score(val_y, pipeline.predict(val_x), average="macro")
    print(f"validation macro F1: {val_macro_f1:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
