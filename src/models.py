"""One predict_proba interface over both model families.

Evaluation and serving load models through `load_predictor` so neither has to
branch on whether it got a TF-IDF pipeline or a fine-tuned transformer.
"""

import json
from pathlib import Path

import numpy as np


class SklearnPredictor:
    """TF-IDF + logistic regression pipeline persisted with joblib."""

    backend = "tfidf+logreg"

    def __init__(self, path: Path):
        import joblib

        self.pipeline = joblib.load(path / "model.joblib")
        self.labels = json.loads((path / "labels.json").read_text())

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)


class TransformerPredictor:
    """Fine-tuned HF sequence classifier, tokenizer loaded from the same dir."""

    backend = "distilbert"

    def __init__(self, path: Path, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device).eval()
        self.labels = json.loads((path / "labels.json").read_text())

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    truncation=True,
                    max_length=64,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits
                out.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out)


def load_predictor(path):
    """Pick the backend from what is actually on disk in `path`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- train a model first")
    return SklearnPredictor(path) if (path / "model.joblib").exists() else TransformerPredictor(path)
