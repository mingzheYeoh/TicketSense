"""Batch evaluation: macro F1, per-class recall, confusion analysis, and the
confidence threshold that the API uses to fall back to manual review.

The threshold is picked on validation only, then reported on test, so the
coverage/accuracy trade-off in the README is an out-of-sample number.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from data import load_splits
from models import load_predictor

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TARGET_ACCURACY = 0.95


def risk_coverage(probs, y):
    """Sweep every possible threshold at once.

    Sorting by confidence descending makes the running mean of `correct` equal
    to "accuracy if we auto-route the k most confident predictions", so there
    is no threshold grid to pick and no operating point that gets skipped.
    Returns (thresholds, coverage, accuracy) aligned element-wise.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y)
    order = np.argsort(-confidence)
    k = np.arange(1, len(order) + 1)
    return confidence[order], k / len(order), np.cumsum(correct[order]) / k


def apply_threshold(probs, y, threshold):
    """Exactly what the API will do: auto-route conf >= threshold, queue the rest."""
    keep = probs.max(axis=1) >= threshold
    correct = probs.argmax(axis=1) == np.asarray(y)
    return float(keep.mean()), float(correct[keep].mean()) if keep.any() else float("nan")


def select_threshold(probs, y, target_accuracy=TARGET_ACCURACY):
    """Lowest confidence bar that still keeps auto-routed accuracy at target.

    Takes the *last* qualifying point rather than the first, to maximise how
    much traffic gets automated -- the first point is usually 1 ticket at 100%.
    The chosen threshold is then re-applied, because ties at the cut admit more
    tickets than the curve index alone implies.
    """
    thresholds, _, accuracy = risk_coverage(probs, y)
    qualifying = np.flatnonzero(accuracy >= target_accuracy)
    threshold = float(thresholds[qualifying[-1]]) if len(qualifying) else 1.0
    coverage, accepted_accuracy = apply_threshold(probs, y, threshold)
    return threshold, coverage, accepted_accuracy


def confusion_report(y_true, y_pred, labels, out_dir, top_n=15):
    """Full matrix to CSV -- 77x77 is unreadable as a heatmap -- plus the pairs that matter."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    ranked = np.argsort(off_diagonal, axis=None)[::-1][:top_n]
    rows = [
        {"true_intent": labels[i], "predicted_intent": labels[j], "count": int(off_diagonal[i, j])}
        for i, j in zip(*np.unravel_index(ranked, off_diagonal.shape))
        if off_diagonal[i, j] > 0
    ]
    pd.DataFrame(rows).to_csv(out_dir / "top_confusions.csv", index=False)
    return rows


def evaluate(name, model_dir, target_accuracy=TARGET_ACCURACY):
    (_, _), (val_x, val_y), (test_x, test_y), labels = load_splits()
    predictor = load_predictor(model_dir)
    out_dir = REPORTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_probs = predictor.predict_proba(val_x)
    threshold, val_coverage, val_accuracy = select_threshold(val_probs, val_y, target_accuracy)

    test_probs = predictor.predict_proba(test_x)
    test_pred = test_probs.argmax(axis=1)
    test_coverage, test_accuracy = apply_threshold(test_probs, test_y, threshold)

    recalls = recall_score(test_y, test_pred, average=None, labels=range(len(labels)), zero_division=0)
    pd.DataFrame({"intent": labels, "recall": recalls}).sort_values("recall").to_csv(
        out_dir / "per_class_recall.csv", index=False
    )
    confusions = confusion_report(test_y, test_pred, labels, out_dir)

    thresholds, coverage, accuracy = risk_coverage(test_probs, test_y)
    pd.DataFrame({"threshold": thresholds, "coverage": coverage, "accuracy": accuracy}).iloc[::10].to_csv(
        out_dir / "risk_coverage.csv", index=False
    )

    metrics = {
        "model": name,
        "backend": predictor.backend,
        "test_accuracy": float((test_pred == np.asarray(test_y)).mean()),
        "test_macro_f1": float(f1_score(test_y, test_pred, average="macro")),
        "test_weighted_f1": float(f1_score(test_y, test_pred, average="weighted")),
        "worst_class_recall": float(recalls.min()),
        "worst_intents": [labels[i] for i in np.argsort(recalls)[:5]],
        "review_threshold": threshold,
        "threshold_target_accuracy": target_accuracy,
        "val_auto_routed_share": val_coverage,
        "val_auto_routed_accuracy": val_accuracy,
        "test_auto_routed_share": test_coverage,
        "test_auto_routed_accuracy": test_accuracy,
        "top_confusions": confusions[:5],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # The API reads the threshold from the model dir, so what gets served can
    # never drift from the evaluation that produced it.
    (Path(model_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


DISPLAY_NAMES = {
    "tfidf+logreg": "TF-IDF + Logistic Regression",
    "distilbert": "DistilBERT (fine-tuned)",
}
# The operating points sit close together on x, so their labels are thrown in
# opposite directions rather than landing on top of each other.
LABEL_OFFSETS = [(-14, -46, "right"), (14, 24, "left")]


def plot_risk_coverage():
    """Combined chart of every model evaluated so far, with each operating point marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = sorted(REPORTS.glob("*/risk_coverage.csv"))
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, path in enumerate(curves):
        metrics = json.loads((path.parent / "metrics.json").read_text())
        curve = pd.read_csv(path)
        (line,) = ax.plot(
            curve["coverage"],
            curve["accuracy"],
            linewidth=2.2,
            label=DISPLAY_NAMES.get(metrics["backend"], metrics["model"]),
        )

        x, y = metrics["test_auto_routed_share"], metrics["test_auto_routed_accuracy"]
        dx, dy, align = LABEL_OFFSETS[index % len(LABEL_OFFSETS)]
        ax.plot(x, y, "o", color=line.get_color(), markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(
            f"{x:.0%} automated\nat conf {metrics['review_threshold']:.2f}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            fontsize=10,
            color=line.get_color(),
            fontweight="bold",
        )

    ax.axhline(TARGET_ACCURACY, color="#555", linestyle="--", linewidth=1.2)
    ax.text(0.02, TARGET_ACCURACY + 0.0025, f"{TARGET_ACCURACY:.0%} quality bar", color="#555", fontsize=10)
    ax.set_xlabel("Share of tickets routed automatically (coverage)")
    ax.set_ylabel("Accuracy on auto-routed tickets")
    ax.set_title("TicketSense - how much support traffic can we automate before quality drops?")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(top=1.006)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "risk_coverage.png", dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="report folder name, e.g. baseline")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    args = parser.parse_args()

    metrics = evaluate(args.name, args.model_dir, args.target_accuracy)
    plot_risk_coverage()

    print(f"\n{args.name} ({metrics['backend']}) on the official test set")
    print(f"  accuracy        {metrics['test_accuracy']:.4f}")
    print(f"  macro F1        {metrics['test_macro_f1']:.4f}")
    print(f"  weighted F1     {metrics['test_weighted_f1']:.4f}")
    print(f"  worst recall    {metrics['worst_class_recall']:.2f}  ({metrics['worst_intents'][0]})")
    print(f"  threshold       {metrics['review_threshold']:.3f} (picked on val)")
    print(
        f"  auto-routed     {metrics['test_auto_routed_share']:.1%} of tickets "
        f"at {metrics['test_auto_routed_accuracy']:.1%} accuracy"
    )
    print("  top confusions:")
    for row in metrics["top_confusions"]:
        print(f"    {row['count']:>3}x  {row['true_intent']} -> {row['predicted_intent']}")


if __name__ == "__main__":
    main()
