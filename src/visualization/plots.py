"""General-purpose, file-saving scientific plots."""

from pathlib import Path

import matplotlib.pyplot as plt


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def distribution(frame, column: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame[column].dropna(), bins=40, color="steelblue", edgecolor="white")
    ax.set(title=f"Distribution of {column}", xlabel=column, ylabel="Count")
    ax.grid(alpha=.25)
    _save(fig, path)


def predicted_vs_actual(actual, predicted, target: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, alpha=.35)
    limits = [min(min(actual), min(predicted)), max(max(actual), max(predicted))]
    ax.plot(limits, limits, "k--", label="Ideal")
    ax.set(title=f"Predicted vs actual {target}", xlabel=f"Actual {target}", ylabel=f"Predicted {target}")
    ax.legend()
    ax.grid(alpha=.25)
    _save(fig, path)
