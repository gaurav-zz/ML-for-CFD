"""Plots specific to aerodynamic coefficient analysis."""

from .plots import _save
import matplotlib.pyplot as plt


def coefficient_vs_alpha(frame, coefficient: str, path, group_column: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if group_column and group_column in frame:
        for _, group in frame.groupby(group_column):
            ax.plot(group["Alpha"], group[coefficient], alpha=.35)
    else:
        ax.scatter(frame["Alpha"], frame[coefficient], s=8, alpha=.3)
    ax.set(title=f"{coefficient} vs angle of attack", xlabel="Angle of attack, Alpha (deg)", ylabel=coefficient)
    ax.grid(alpha=.25)
    _save(fig, path)


def airfoil_geometry(coordinates, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(coordinates[:, 0], coordinates[:, 1])
    ax.set(title="Airfoil geometry", xlabel="x/c", ylabel="y/c", aspect="equal")
    ax.grid(alpha=.25)
    _save(fig, path)
