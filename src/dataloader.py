from datasets import DatasetDict, load_dataset
from utils.config import CONFIG

DATASET_NAME = CONFIG["dataset"]["name"]
DATASET_SUBSET = CONFIG["dataset"]["subset"]
TEXT_COLUMN = CONFIG["dataset"]["text_column"]
LABEL_COLUMN = CONFIG["dataset"].get("label_column")  # None/missing -> auto-detect

# Column names commonly used for the target, checked in this order (case-insensitive)
COMMON_LABEL_NAMES = ("label", "labels", "y", "target", "class", "category",
                      "sentiment", "intent", "topic")
ID_LIKE_NAMES = ("id", "idx", "index", "uid", "guid")
LABEL_DTYPES = ("bool", "string", "large_string",
                "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
# Other names datasets use for the validation split
VALIDATION_ALIASES = ("valid", "val", "dev", "eval", "validation_matched")


def get_dataset(seed=42):
    """Load the dataset from config and return it with train/validation/test splits."""
    print(f"Loading dataset '{DATASET_NAME}' (subset: '{DATASET_SUBSET}')...")
    # datasets 5.x no longer supports dataset loading scripts, so we load the
    # Hub's official parquet export (refs/convert/parquet) — identical data.
    dataset = load_dataset(
        DATASET_NAME,
        revision="refs/convert/parquet",
        data_dir=DATASET_SUBSET,  # language folder, e.g. "english"
    )
    # Keep existing splits, create any missing ones from train
    return ensure_splits(dataset, find_label_column(dataset), seed=seed)


def find_label_column(dataset, label_column=None, split="train", max_classes=100):
    """
    Identify the target (y) column of a classification dataset.

    Order of checks:
      1. label_column given (argument, or "label_column" in config.json) -> use it
      2. exactly one ClassLabel column -> that's the target
      3. a column with a common target name (label, y, target, class, ...)
      4. the only low-cardinality int/str/bool column that isn't the text or an id
    Raises ValueError if the target can't be identified unambiguously.
    """
    data = dataset[split]
    features = data.features

    label_column = label_column or LABEL_COLUMN
    if label_column:
        if label_column not in features:
            raise ValueError(f"Label column '{label_column}' not found. Columns: {list(features)}")
        return label_column

    class_label_cols = [c for c, f in features.items() if hasattr(f, "names")]
    if len(class_label_cols) == 1:
        return class_label_cols[0]

    lower_to_col = {c.lower(): c for c in features}
    for name in COMMON_LABEL_NAMES:
        if name in lower_to_col:
            return lower_to_col[name]

    candidates = []
    for col, feature in features.items():
        if col == TEXT_COLUMN or col.lower() in ID_LIKE_NAMES:
            continue
        if getattr(feature, "dtype", None) not in LABEL_DTYPES:
            continue
        n_unique = len(set(data[col]))
        if 1 < n_unique <= max_classes and n_unique < 0.5 * len(data):
            candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Several possible label columns {candidates}; set 'label_column' in config.json")
    raise ValueError(f"No label column found in {list(features)}; set 'label_column' in config.json")


def get_label_names(dataset, label_column=None, split="train"):
    """
    Return the class names for the target column (auto-detected if not given).

    Uses the ClassLabel names when the dataset defines them (in label-id order),
    otherwise the sorted unique values found in the given split.
    """
    label_column = label_column or find_label_column(dataset, split=split)
    data = dataset[split]
    feature = data.features[label_column]
    if hasattr(feature, "names"):
        return list(feature.names)
    return [str(v) for v in sorted(set(data[label_column]))]


def _split_off(data, size, label_column, seed):
    """Split `size` (fraction) off `data`, stratified by label when possible. Returns (rest, split)."""
    stratify = label_column if hasattr(data.features[label_column], "names") else None
    try:
        parts = data.train_test_split(test_size=size, seed=seed, stratify_by_column=stratify)
    except ValueError:
        # e.g. a class with too few examples to stratify -> plain random split
        parts = data.train_test_split(test_size=size, seed=seed)
    return parts["train"], parts["test"]


def ensure_splits(dataset, label_column=None, val_size=0.1, test_size=0.1, seed=42):
    """
    Make sure the dataset has 'train', 'validation' and 'test' splits.

    Existing splits are kept as they are. Missing ones are carved out of train
    (stratified by label when the label is a ClassLabel, with a fixed seed so
    the result is the same on every run). val_size/test_size are fractions of
    the original train split. A test split whose labels are all -1 (hidden
    labels, e.g. GLUE) counts as missing.
    """
    if "train" not in dataset:
        raise ValueError(f"Dataset has no 'train' split. Splits: {list(dataset)}")

    splits = dict(dataset)
    label_column = label_column or find_label_column(dataset)

    # Use an alternative validation split name if present (dev, valid, ...)
    if "validation" not in splits:
        for alias in VALIDATION_ALIASES:
            if alias in splits:
                print(f"Using '{alias}' split as validation")
                splits["validation"] = splits.pop(alias)
                break

    if "test" in splits and all(v == -1 for v in splits["test"][label_column]):
        print("Test split has hidden labels (-1); creating a labelled test split from train")
        del splits["test"]

    train = splits["train"]
    n_train = len(train)

    if "test" not in splits:
        train, splits["test"] = _split_off(train, test_size, label_column, seed)
        print(f"Created 'test' split from train ({len(splits['test'])} rows)")

    if "validation" not in splits:
        # Rescale so validation is val_size of the *original* train size
        train, splits["validation"] = _split_off(train, val_size * n_train / len(train),
                                                 label_column, seed)
        print(f"Created 'validation' split from train ({len(splits['validation'])} rows)")

    splits["train"] = train
    return DatasetDict({name: splits[name] for name in ("train", "validation", "test")})
