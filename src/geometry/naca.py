"""Standard NACA four-digit airfoil geometry generation."""

from __future__ import annotations

import numpy as np


def parse_naca4(code: str) -> tuple[float, float, float]:
    code = str(code).upper().removeprefix("NACA").strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must contain exactly four digits, e.g. '2412'.")
    return int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100


def generate_naca4(code: str, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return upper and lower surfaces ordered from leading to trailing edge."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    camber, position, thickness = parse_naca4(code)
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    if camber == 0 or position == 0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < position, camber / position**2 * (2 * position * x - x**2), camber / (1-position)**2 * ((1-2*position) + 2*position*x - x**2))
        dyc = np.where(x < position, 2*camber / position**2 * (position-x), 2*camber / (1-position)**2 * (position-x))
    theta = np.arctan(dyc)
    upper = np.column_stack((x - yt*np.sin(theta), yc + yt*np.cos(theta)))
    lower = np.column_stack((x + yt*np.sin(theta), yc - yt*np.cos(theta)))
    return upper, lower
