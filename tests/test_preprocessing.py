import numpy as np
import pandas as pd

from src.data.preprocessing import fit_transform_features, geometry_aware_split


def test_geometry_aware_split_keeps_groups_together():
    frame = pd.DataFrame({"value": range(20)})
    groups = pd.Series([f"foil-{i // 4}" for i in range(20)])
    split = geometry_aware_split(frame, groups, seed=1)
    partitions = {id_: name for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)) for id_ in groups.loc[part.index].unique()}
    assert len(partitions) == 5
    for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        assert all(partitions[group] == name for group in groups.loc[part.index].unique())


def test_scaler_fits_training_only():
    train = np.array([[0.0], [2.0]])
    validation = np.array([[100.0]])
    transformed_train, transformed_validation, _, scaler = fit_transform_features(train, validation, validation)
    assert scaler.mean_[0] == 1.0
    assert transformed_train.mean() == 0.0
    assert transformed_validation[0, 0] > 10
