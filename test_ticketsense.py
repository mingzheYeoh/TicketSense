"""Self-checks for the logic that can break quietly.

Run: python test_ticketsense.py
Deliberately offline and model-free -- threshold selection is the piece where a
bug produces plausible-looking numbers instead of a crash.
"""

import sys
from pathlib import Path

import numpy as np

# Resolved from this file, not the cwd, so the checks run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluate import apply_threshold, risk_coverage, select_threshold  # noqa: E402

# Three classes, not two: confidence is max(probs), so a 2-class fixture can
# never express a prediction less confident than 0.5.
N_CLASSES = 3


def probs_from(rows):
    """rows: [(confidence, predicted_class)]. Remaining mass is split evenly."""
    probs = np.empty((len(rows), N_CLASSES))
    for i, (confidence, predicted) in enumerate(rows):
        probs[i] = (1 - confidence) / (N_CLASSES - 1)
        probs[i, predicted] = confidence
    return probs


def test_risk_coverage_is_sorted_by_confidence():
    thresholds, coverage, accuracy = risk_coverage(
        probs_from([(0.6, 1), (0.9, 1), (0.7, 1)]), [1, 1, 1]
    )
    assert list(thresholds) == [0.9, 0.7, 0.6], thresholds
    assert np.allclose(coverage, [1 / 3, 2 / 3, 1.0])
    assert np.allclose(accuracy, [1.0, 1.0, 1.0])


def test_threshold_trades_coverage_for_accuracy():
    # The three least confident predictions are the wrong ones.
    probs = probs_from([(0.99, 1), (0.95, 1), (0.9, 1), (0.6, 0), (0.55, 0), (0.5, 0)])
    threshold, coverage, accuracy = select_threshold(probs, [1] * 6, target_accuracy=0.95)
    assert accuracy == 1.0
    assert coverage == 0.5, coverage
    assert threshold == 0.9, threshold


def test_threshold_maximises_coverage_not_just_accuracy():
    # A naive "first qualifying point" pick would stop at 1 example / 100%.
    probs = probs_from([(0.99, 1), (0.98, 1), (0.97, 1), (0.96, 1), (0.4, 0)])
    _, coverage, _ = select_threshold(probs, [1] * 5, target_accuracy=0.95)
    assert coverage == 0.8, coverage


def test_unreachable_target_routes_everything_to_humans():
    probs = probs_from([(0.9, 1), (0.8, 1), (0.7, 1)])
    threshold, coverage, _ = select_threshold(probs, [0, 0, 0], target_accuracy=0.95)
    assert threshold == 1.0 and coverage == 0.0


def test_apply_threshold_admits_confidence_ties():
    # Ties at the cut must all be admitted, or the API would route fewer
    # tickets than the evaluation promised.
    coverage, accuracy = apply_threshold(probs_from([(0.8, 1), (0.8, 1), (0.8, 0)]), [1, 1, 1], 0.8)
    assert coverage == 1.0 and np.isclose(accuracy, 2 / 3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
