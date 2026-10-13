import numpy as np
import pytest

from src.training.evaluate import evaluate_model
from src.training.targets import (
    DEFAULT_TARGET_TRANSFORMS,
    INVERSE_LIMITS,
    PhysicalPredictor,
    forward_targets,
    inverse_targets,
    resolve_transforms,
    transform_names,
)

LOG_CD = {"Cd": "log"}


class Fixed:
    """Predicts in fitting space, so the wrapper must invert before scoring."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, x):
        return self.values


def test_nothing_is_transformed_unless_the_caller_asks():
    """A silent log on Cd would change every metric without appearing anywhere."""
    assert transform_names(["Cl", "Cd", "Cm"]) == {
        "Cl": "identity", "Cd": "identity", "Cm": "identity",
    }
    np.testing.assert_allclose(forward_targets(np.array([0.01, 0.5]), ["Cd"])[:, 0], [0.01, 0.5])


def test_the_recommended_mapping_logs_drag_only():
    assert DEFAULT_TARGET_TRANSFORMS == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    names = ["Cl", "Cd", "Cm"]
    assert transform_names(names, DEFAULT_TARGET_TRANSFORMS) == DEFAULT_TARGET_TRANSFORMS
    chosen = resolve_transforms(["Cd", "Cl"], DEFAULT_TARGET_TRANSFORMS)
    assert [transform.name for transform in chosen] == ["log", "identity"]


def test_forward_and_inverse_round_trip_in_physical_units():
    y = np.column_stack([
        np.array([-1.2, 0.0, 1.8]),        # Cl spans zero
        np.array([0.00265, 0.012, 0.59]),  # Cd spans the archive's 223x range
        np.array([-0.08, 0.0, 0.05]),      # Cm spans zero
    ])
    names = ["Cl", "Cd", "Cm"]
    fitting = forward_targets(y, names, DEFAULT_TARGET_TRANSFORMS)
    # Only the drag column is transformed.
    np.testing.assert_allclose(fitting[:, [0, 2]], y[:, [0, 2]])
    np.testing.assert_allclose(fitting[:, 1], np.log(y[:, 1]))
    np.testing.assert_allclose(inverse_targets(fitting, names, DEFAULT_TARGET_TRANSFORMS), y, rtol=1e-12)


def test_logging_drag_compresses_its_dynamic_range():
    drag = np.array([0.00265, 0.59])
    logged = forward_targets(drag, ["Cd"], LOG_CD)[:, 0]
    assert drag.max() / drag.min() > 200
    assert abs(logged[1] - logged[0]) < 6


@pytest.mark.parametrize("bad", [[0.0], [-0.01], [np.nan], [np.inf]])
def test_non_positive_drag_is_refused_with_an_actionable_message(bad):
    with pytest.raises(ValueError, match="positive|non-finite"):
        forward_targets(np.array(bad), ["Cd"], LOG_CD)


def test_inverse_clamps_runaway_predictions_instead_of_overflowing():
    """An extrapolating model can emit a log-drag of +400; exp would be inf."""
    recovered = inverse_targets(np.array([[-5000.0], [5000.0]]), ["Cd"], LOG_CD)
    assert np.isfinite(recovered).all()
    np.testing.assert_allclose(recovered[:, 0], list(INVERSE_LIMITS))


def test_physical_predictor_reports_metrics_in_coefficient_units():
    actual = np.column_stack([[1.0, 2.0], [0.01, 0.02]])
    # The wrapped model emits log-drag, so raw scoring would be nonsense.
    model = Fixed(np.column_stack([[1.0, 2.0], np.log([0.011, 0.022])]))
    wrapped = PhysicalPredictor(model, ["Cl", "Cd"], DEFAULT_TARGET_TRANSFORMS)
    np.testing.assert_allclose(wrapped.predict(None), np.column_stack([[1.0, 2.0], [0.011, 0.022]]))

    scores = evaluate_model(wrapped, np.zeros((2, 1)), actual, target_names=["Cl", "Cd"])
    assert scores["targets"]["Cl"]["mae"] == pytest.approx(0.0)
    assert scores["targets"]["Cd"]["mae"] == pytest.approx(0.0015, rel=1e-9)
    # 10% over on both drag rows, which an absolute MAE of 0.0015 would hide.
    assert scores["targets"]["Cd"]["median_relative_error"] == pytest.approx(0.1, rel=1e-9)
    assert "log" in repr(wrapped)


def test_physical_predictor_keeps_a_single_target_one_dimensional():
    wrapped = PhysicalPredictor(Fixed(np.log([[0.01], [0.02]])), ["Cd"], LOG_CD)
    prediction = wrapped.predict(None)
    assert prediction.shape == (2,)
    np.testing.assert_allclose(prediction, [0.01, 0.02])


@pytest.mark.parametrize("names,mapping,error", [
    (["Cl"], {"Cl": "sqrt"}, "Unknown transform"),
    (["Cl"], {"Cx": "log"}, "not in use"),
    ([], None, "nonempty sequence"),
    ("Cl", None, "nonempty sequence"),
])
def test_invalid_transform_requests_are_rejected(names, mapping, error):
    with pytest.raises(ValueError, match=error):
        resolve_transforms(names, mapping)


def test_a_target_the_mapping_omits_is_fitted_as_given():
    assert transform_names(["Cl", "Cd"], LOG_CD) == {"Cl": "identity", "Cd": "log"}


@pytest.mark.parametrize("values", [np.zeros((3, 3)), np.zeros((3, 1))])
def test_column_count_must_match_the_target_names(values):
    with pytest.raises(ValueError, match="one column per target"):
        forward_targets(values, ["Cl", "Cd"])
