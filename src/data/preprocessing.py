"""Leakage-safe splitting and feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def geometry_aware_split(frame: pd.DataFrame, groups: pd.Series, seed: int = 42, train_size: float = .7, validation_size: float = .15) -> SplitData:
    """Split complete geometries into train, validation, and test partitions."""
    if len(frame) != len(groups):
        raise ValueError("Groups must have one value per row.")
    first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, remainder_idx = next(first.split(frame, groups=groups))
    remainder = frame.iloc[remainder_idx]
    remainder_groups = groups.iloc[remainder_idx]
    relative_validation = validation_size / (1 - train_size)
    second = GroupShuffleSplit(n_splits=1, train_size=relative_validation, random_state=seed)
    validation_rel, test_rel = next(second.split(remainder, groups=remainder_groups))
    return SplitData(frame.iloc[train_idx].copy(), remainder.iloc[validation_rel].copy(), remainder.iloc[test_rel].copy())


def fit_transform_features(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a scaler on training features only and transform all partitions."""
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(validation), scaler.transform(test), scaler
