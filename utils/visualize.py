from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.config import PROJECT_ROOT

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Theme from the notebook's class_distribution.png
SENTIMENT_COLORS = {
    "negative": "#ff6b6b",
    "neutral": "#95a5a6",
    "no_impact": "#95a5a6",
    "positive": "#51cf66",
    "mixed": "#ffa94d",
}
# Colors for any other class names, assigned in this fixed order
EXTRA_COLORS = ["#4dabf7", "#ffa94d", "#845ef7", "#f783ac", "#20c997", "#ffd43b", "#868e96"]


def get_class_colors(label_names):
    """One color per class: sentiment names get their fixed color, others the next free extra color."""
    extras = iter(c for c in EXTRA_COLORS if c not in [SENTIMENT_COLORS.get(n.lower()) for n in label_names])
    return [SENTIMENT_COLORS.get(name.lower()) or next(extras, "#adb5bd") for name in label_names]


def count_classes(split_data, label_column, label_names):
    """Counts per class (in label_names order) for one split; works for ClassLabel and plain labels."""
    is_class_label = hasattr(split_data.features[label_column], "names")
    counts = Counter(label_names[v] if is_class_label else str(v) for v in split_data[label_column])
    return [counts.get(name, 0) for name in label_names]


def _style_bar_axis(ax, title, label_names):
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=30 if len(label_names) > 4 else 0)


def _label_bars(ax, bars, counts):
    total = sum(counts)
    for bar, count in zip(bars, counts):
        share = 100 * count / total if total else 0
        ax.annotate(f"{count}\n({share:.1f}%)", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    ax.margins(y=0.18)  # room for the labels above the tallest bar


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"📊 Saved '{path}'")


def plot_class_distribution(dataset, label_column, label_names, output_dir=OUTPUTS_DIR, show=False):
    """
    Save two class distribution charts:
      - class_distribution.png:           all splits combined (bar + pie)
      - class_distribution_by_split.png:  one bar chart per split (train / validation / test)
    """
    colors = get_class_colors(label_names)
    split_counts = {split: count_classes(dataset[split], label_column, label_names) for split in dataset}
    total_counts = [sum(c) for c in zip(*split_counts.values())]

    # 1. Aggregated: bar + pie over all splits
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(label_names, total_counts, color=colors, edgecolor="black")
    _label_bars(axes[0], bars, total_counts)
    _style_bar_axis(axes[0], f"Class Distribution (all splits, n={sum(total_counts)})", label_names)

    shown = [(n, c, col) for n, c, col in zip(label_names, total_counts, colors) if c > 0]
    names, sizes, pie_colors = zip(*shown)
    axes[1].pie(sizes, labels=names, autopct="%1.1f%%", colors=pie_colors,
                explode=[0.05] * len(sizes), startangle=90)
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(output_dir) / "class_distribution.png")

    # 2. Per split: same theme, one bar chart per split. Each panel has its own
    # y-scale (splits differ a lot in size); the % labels make them comparable.
    splits = list(split_counts)
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        counts = split_counts[split]
        bars = ax.bar(label_names, counts, color=colors, edgecolor="black")
        _label_bars(ax, bars, counts)
        _style_bar_axis(ax, f"{split.capitalize()} (n={sum(counts)})", label_names)
    fig.suptitle("Class Distribution per Split", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(output_dir) / "class_distribution_by_split.png")

    if show:
        plt.show()  # shows both figures at once
    plt.close("all")


def plot_token_lengths(lengths, max_length, output_dir=OUTPUTS_DIR, show=False, split="train"):
    """Save a histogram of token lengths with the max_length cut-off marked (token_lengths.png)."""
    truncated = (lengths > max_length).mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=min(50, len(set(lengths))), color="#4dabf7", edgecolor="black")
    ax.axvline(max_length, color="#ff6b6b", linewidth=2.5)
    ax.annotate(f"max_length = {max_length}\n{truncated:.1f}% truncated",
                (max_length, 1), xycoords=("data", "axes fraction"), xytext=(-8, -8),
                textcoords="offset points", ha="right", va="top", fontsize=10, fontweight="bold")
    ax.axvline(lengths.mean(), color="#495057", linewidth=1.5)
    ax.annotate(f"mean = {lengths.mean():.1f}", (lengths.mean(), 0.75), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=10)

    ax.set_xlim(0, max(max_length, lengths.max()) * 1.05)
    ax.set_title(f"Token Length Distribution ({split}, n={len(lengths)})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tokens per example", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "token_lengths.png")

    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, label_names, output_dir=OUTPUTS_DIR, show=False,
                          split="test"):
    """
    Save a confusion matrix (confusion_matrix.png): rows = true class, columns = predicted.
    Cells show the count and the share of that true class (row %), shaded by row %.
    """
    n = len(label_names)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_share = np.divide(matrix, row_totals, out=np.zeros(matrix.shape), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(1.6 * n + 3, 1.4 * n + 2))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            color = "white" if row_share[i, j] > 0.6 else "black"
            text = f"{matrix[i, j]}\n({100 * row_share[i, j]:.0f}%)" if row_totals[i, 0] else "–"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(n), label_names, rotation=30 if n > 4 else 0)
    ax.set_yticks(range(n), [f"{name} (n={row_totals[i, 0]})" for i, name in enumerate(label_names)])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    accuracy = np.trace(matrix) / max(matrix.sum(), 1)
    ax.set_title(f"Confusion Matrix ({split}, accuracy {accuracy:.1%})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, Path(output_dir) / "confusion_matrix.png")

    if show:
        plt.show()
    plt.close(fig)
