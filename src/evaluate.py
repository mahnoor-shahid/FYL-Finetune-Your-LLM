import json

import numpy as np
from sklearn.metrics import classification_report

from utils.config import PROJECT_ROOT
from utils.visualize import plot_confusion_matrix

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def evaluate_on_test(trainer, tokenized_datasets, label_names, split="test", show_plots=False):
    """
    Final evaluation on the held-out test split (never used for training or model selection).

    Prints and saves a per-class classification report, saves the metrics as JSON
    and plots the confusion matrix.
    """
    print(f"\n🧪 Evaluating on {split} ({len(tokenized_datasets[split])} examples)...")
    output = trainer.predict(tokenized_datasets[split], metric_key_prefix=split)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = output.label_ids

    # Report only classes that occur in this split (others have no meaningful recall/F1),
    # so the macro average matches compute_metrics
    present = [i for i in range(len(label_names)) if (y_true == i).any()]
    report = classification_report(y_true, y_pred, labels=present,
                                   target_names=[label_names[i] for i in present],
                                   digits=4, zero_division=0)
    print(report)

    missing = [name for i, name in enumerate(label_names) if i not in present]
    if missing:
        n_predicted = {name: int((y_pred == label_names.index(name)).sum()) for name in missing}
        print(f"⚠️ No {split} examples for: {', '.join(missing)} -> left out of the averages "
              f"(times predicted anyway: {n_predicted})")

    metrics = {k: float(v) for k, v in output.metrics.items()}
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / f"{split}_report.txt").write_text(report, encoding="utf-8")
    (OUTPUTS_DIR / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"💾 Report and metrics saved to '{OUTPUTS_DIR}'")

    plot_confusion_matrix(y_true, y_pred, label_names, show=show_plots, split=split)
    return metrics
