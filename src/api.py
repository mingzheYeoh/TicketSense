"""FastAPI service: classify a support ticket, or send it to a human.

Every response carries the confidence and a `needs_review` flag. The threshold
behind that flag is not a hand-picked constant -- it is read from the
metrics.json that evaluate.py writes next to the model, so what the service
does in production is exactly what was measured on validation data.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from models import load_predictor

MODEL_DIR = Path(os.getenv("TICKETSENSE_MODEL", "models/distilbert-v1"))
TOP_K = 3

state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"{metrics_path} is missing. Run evaluate.py for this model first -- "
            "serving without a validation-selected threshold would silently "
            "auto-route tickets the model is not confident about."
        )
    state["metrics"] = json.loads(metrics_path.read_text())
    state["predictor"] = load_predictor(MODEL_DIR)
    yield
    state.clear()


app = FastAPI(
    title="TicketSense",
    description="Banking support intent classification with a manual-review fallback.",
    version="1.0.0",
    lifespan=lifespan,
)


class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=2000, examples=["My card still hasn't arrived"])


class TicketBatch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class Scored(BaseModel):
    intent: str
    confidence: float


class Prediction(Scored):
    needs_review: bool
    alternatives: list[Scored]


def score(texts: list[str]) -> list[Prediction]:
    predictor, threshold = state["predictor"], state["metrics"]["review_threshold"]
    probs = predictor.predict_proba(texts)
    ranked = probs.argsort(axis=1)[:, ::-1][:, :TOP_K]
    return [
        Prediction(
            intent=predictor.labels[row[0]],
            confidence=round(float(probs[i, row[0]]), 4),
            needs_review=bool(probs[i, row[0]] < threshold),
            alternatives=[
                Scored(intent=predictor.labels[j], confidence=round(float(probs[i, j]), 4))
                for j in row[1:]
            ],
        )
        for i, row in enumerate(ranked)
    ]


@app.get("/health")
def health():
    metrics = state["metrics"]
    return {
        "status": "ok",
        "model": metrics["model"],
        "backend": metrics["backend"],
        "intents": len(state["predictor"].labels),
        "review_threshold": metrics["review_threshold"],
        "expected_auto_routed_share": round(metrics["test_auto_routed_share"], 4),
        "expected_auto_routed_accuracy": round(metrics["test_auto_routed_accuracy"], 4),
    }


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket):
    return score([ticket.text])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: TicketBatch):
    return score(batch.texts)
