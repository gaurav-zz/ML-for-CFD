import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import fit_transform_features, geometry_aware_split


@pytest.mark.parametrize("n_groups,train_size,validation_size,expected_counts", [
    (5, .7, .15, (3, 1, 1)),
    (3, .5, .25, (1, 1, 1)),
    (20, .7, .15, (14, 3, 3)),
    (5, .4, .2, (2, 1, 2)),
    (81, .95, .03, (76, 3, 2)),
    (20, .7, .149999, (14, 2, 4)),
    (4, .5, .4999999999999999, (2, 1, 1)),
])
def test_geometry_aware_split_keeps_groups_together(n_groups, train_size, validation_size, expected_counts):
    frame = pd.DataFrame({"value": range(n_groups * 4)}, index=np.arange(n_groups * 4) + 100)
    groups = pd.Series([f"foil-{i // 4}" for i in range(len(frame))], index=frame.index)
    split = geometry_aware_split(frame, groups, seed=1, train_size=train_size, validation_size=validation_size)
    parts = (split.train, split.validation, split.test)
    assert tuple(groups.loc[part.index].nunique() for part in parts) == expected_counts
    partitions = {id_: name for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)) for id_ in groups.loc[part.index].unique()}
    assert len(partitions) == n_groups
    for name, part in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        assert all(partitions[group] == name for group in groups.loc[part.index].unique())
    pd.testing.assert_frame_equal(pd.concat(parts).sort_index(), frame)

    repeated = geometry_aware_split(frame, groups, seed=1, train_size=train_size, validation_size=validation_size)
    for part, repeated_part in zip(parts, (repeated.train, repeated.validation, repeated.test)):
        pd.testing.assert_frame_equal(part, repeated_part)


@pytest.mark.parametrize("train_size,validation_size", [
    (0, .15), (-.1, .15), (1, .15), (.7, 0), (.7, -.1), (.7, .3), (np.nan, .15), (.7, np.inf),
])
def test_geometry_aware_split_rejects_invalid_fractions(train_size, validation_size):
    frame = pd.DataFrame({"value": range(5)})
    with pytest.raises(ValueError, match="positive and sum to less than 1"):
        geometry_aware_split(frame, pd.Series(range(5)), train_size=train_size, validation_size=validation_size)


@pytest.mark.parametrize("groups,message", [
    ([0, 1, 2, 3], "one value per row"),
    ([0, 1, 2, 3, None], "must not contain missing values"),
])
def test_geometry_aware_split_rejects_incomplete_groups(groups, message):
    frame = pd.DataFrame({"value": range(5)})
    with pytest.raises(ValueError, match=message):
        geometry_aware_split(frame, pd.Series(groups))


@pytest.mark.parametrize("n_groups,validation_size", [(3, .15), (5, .1)])
def test_geometry_aware_split_rejects_empty_partitions(n_groups, validation_size):
    frame = pd.DataFrame({"value": range(n_groups)})
    with pytest.raises(ValueError, match="nonempty validation and test"):
        geometry_aware_split(frame, pd.Series(range(n_groups)), validation_size=validation_size)


def test_scaler_fits_training_only():
    train = np.array([[0.0], [2.0]])
    validation = np.array([[100.0]])
    transformed_train, transformed_validation, _, scaler = fit_transform_features(train, validation, validation)
    assert scaler.mean_[0] == 1.0
    assert transformed_train.mean() == 0.0
    assert transformed_validation[0, 0] > 10
