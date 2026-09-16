"""Fine-tune DistilBERT on BANKING77.

Model and tokenizer are saved together into a versioned directory -- serving a
transformer with a tokenizer it was not trained with fails silently, producing
confident nonsense rather than an error.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import SEED, load_splits

BASE_MODEL = "distilbert-base-uncased"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-v1"
MAX_LENGTH = 64  # p99 of BANKING77 is well under this; longer just wastes compute
EPOCHS = 10  # val macro F1 was still climbing at 6
BATCH_SIZE = 32


class TicketDataset:
    """Minimal torch-style dataset -- Trainer only needs __len__ and __getitem__."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    (train_x, train_y), (val_x, val_y), _, labels = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode(texts):
        return tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)

    train_ds = TicketDataset(encode(train_x), list(train_y))
    val_ds = TicketDataset(encode(val_x), list(val_y))

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id={name: i for i, name in enumerate(labels)},
    )

    steps_per_epoch = -(-len(train_x) // BATCH_SIZE)  # ceil
    args = TrainingArguments(
        output_dir=str(MODEL_DIR.parent / "_distilbert_checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=128,
        # transformers 5 dropped warmup_ratio; 10% of total steps, same idea.
        warmup_steps=int(0.1 * steps_per_epoch * EPOCHS),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=125,
        disable_tqdm=True,  # logging_steps already reports progress; tqdm just floods CI logs
        fp16=True,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("best validation:", trainer.evaluate())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
