"""BANKING77 loading and the one train/val split every script shares.

Source is the `mteb/banking77` mirror: the original `PolyAI/banking77` repo
still ships a loading script, which datasets>=4 refuses to execute. The mirror
is the same corpus with a handful of exact duplicates removed
(9,993 train / 3,076 test vs the 10,003 / 3,080 quoted in the paper).

The official test split is never used for tuning, thresholds or model
selection -- validation is carved out of the training set instead, so the
numbers in the README are what the model would actually do on unseen tickets.
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRACTION = 0.2
DATASET = "mteb/banking77"


def _label_names(dataset):
    """Recover index -> intent name.

    The mirror stores `label` as a plain int, so the mapping has to be rebuilt
    from `label_text`. Getting this wrong would rename every row of every
    per-class report while leaving the numbers looking perfectly reasonable,
    so it is checked against both splits rather than trusted.
    """
    mapping = {}
    for split in dataset.values():
        for index, name in zip(split["label"], split["label_text"]):
            if mapping.setdefault(index, name) != name:
                raise ValueError(f"label {index} maps to both {mapping[index]!r} and {name!r}")
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError(f"label ids are not a contiguous range: {sorted(mapping)[:5]}...")
    return [mapping[i] for i in range(len(mapping))]


def load_splits():
    """Return ((train_x, train_y), (val_x, val_y), (test_x, test_y), label_names)."""
    dataset = load_dataset(DATASET)
    labels = _label_names(dataset)

    train_x, val_x, train_y, val_y = train_test_split(
        dataset["train"]["text"],
        dataset["train"]["label"],
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=dataset["train"]["label"],  # 77 intents, ~26 val examples each
    )
    return (
        (train_x, train_y),
        (val_x, val_y),
        (dataset["test"]["text"], dataset["test"]["label"]),
        labels,
    )


if __name__ == "__main__":
    (train_x, _), (val_x, _), (test_x, _), labels = load_splits()
    print(f"{len(labels)} intents | train {len(train_x)} | val {len(val_x)} | test {len(test_x)}")
    print(f"first three intents: {labels[:3]}")
