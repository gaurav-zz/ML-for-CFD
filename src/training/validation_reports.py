"""Small, immutable validation reports for epoch-controlled MLP experiments."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.training.comparison import model_artifact_path


def write_validation_reports(output_dir, validation_frame, feature_columns, target_names):
    """Plot completed validation results without fitting models or using test data."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    report = json.loads((output / "run.json").read_text())
    metadata = report["run_metadata"]
    if metadata.get("experiment_type") != "mlp_validation" or report["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("A completed validation-controlled MLP experiment is required.")
    if list(feature_columns) != metadata["feature_columns"] or list(target_names) != report["target_names"]:
        raise ValueError("Report feature and target columns must match the saved experiment.")
    if (
        len(validation_frame) != metadata["split_rows"]["validation"]
        or set(validation_frame.geometry_id) != set(metadata["split_geometries"]["validation"])
    ):
        raise ValueError("Report data must match the saved validation geometries, not training or test.")
    selected = report["selected_models"]
    artifacts = {name: model_artifact_path(report, name) for name in selected.values()}
    metrics = pd.read_csv(output / "comparison.csv")
    by_geometry = pd.read_csv(output / "validation_by_geometry.csv", dtype={"geometry_id": str})
    history = pd.read_csv(output / "validation_history.csv")
    summary_path = output / "validation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("evaluation_scope") != "validation" or summary.get("selected_models") != selected:
            raise ValueError("Existing validation summary does not match the experiment.")
    else:
        grouped = by_geometry.groupby(["model", "target"], sort=False).agg(
            geometry_count=("geometry_id", "nunique"), mean_geometry_mae=("mae", "mean"),
            mean_geometry_rmse=("rmse", "mean"), median_geometry_rmse=("rmse", "median"),
            worst_geometry_rmse=("rmse", "max"),
        ).reset_index()
        summary = {
            "evaluation_scope": "validation", "selected_models": selected,
            "selection_metric": "validation_rmse", "test_used_for_selection": False,
            "checkpoint_metric": "mean_validation_mse_in_train_standardized_target_units",
            "validation_rows": len(validation_frame),
            "validation_geometry_count": int(validation_frame.geometry_id.nunique()),
            "per_geometry_summary": grouped.to_dict("records"),
        }
        temporary = output / "validation_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(summary_path)

    def panels():
        return plt.subplots(1, len(target_names), figsize=(6 * len(target_names), 4), squeeze=False)

    def save(figure, filename):
        try:
            figure.tight_layout()
            temporary = output / f"{filename}.tmp"
            figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
            temporary.replace(output / filename)
        finally:
            plt.close(figure)

    if not (output / "validation_rmse.png").exists():
        fig, axes = panels()
        for index, target in enumerate(target_names):
            values = metrics[metrics.target == target].sort_values(["rmse", "model"])
            axis = axes[0, index]
            axis.barh(values.model, values.rmse)
            axis.invert_yaxis()
            axis.set(title=f"{target}: restored-checkpoint validation RMSE", xlabel="RMSE (original units)")
        save(fig, "validation_rmse.png")

    if not (output / "training_loss.png").exists():
        fig, axis = plt.subplots(figsize=(8, 4))
        for name, result in report["results"].items():
            if result["status"] == "success":
                training = result["training"]
                axis.plot(range(1, len(training["loss_curve"]) + 1), training["loss_curve"], label=name)
        axis.set(xlabel="Epoch", ylabel="Training objective (scaled targets)", title="Training loss, not validation accuracy")
        axis.legend()
        axis.grid(alpha=.3)
        save(fig, "training_loss.png")

    if not (output / "validation_curves.png").exists():
        fig, axes = panels()
        for index, target in enumerate(target_names):
            axis = axes[0, index]
            for name, result in report["results"].items():
                if result["status"] != "success":
                    continue
                values = history[(history.model == name) & (history.target == target)]
                line, = axis.plot(values.epoch, values.rmse, label=name)
                checkpoint = values[values.epoch == result["training"]["best_epoch"]]
                axis.scatter(checkpoint.epoch, checkpoint.rmse, color=line.get_color(), marker="D", s=35)
            axis.set(title=f"{target}: held-out-airfoil validation", xlabel="Epoch (diamonds: joint checkpoints)", ylabel="RMSE (original units)")
            axis.legend()
            axis.grid(alpha=.3)
        save(fig, "validation_curves.png")

    if not (output / "worst_airfoils.png").exists():
        fig, axes = panels()
        for index, target in enumerate(target_names):
            values = by_geometry[(by_geometry.model == selected[target]) & (by_geometry.target == target)].nlargest(10, "rmse")
            axis = axes[0, index]
            axis.barh(values.geometry_id, values.rmse)
            axis.invert_yaxis()
            axis.set(title=f"{target}: worst validation airfoils", xlabel="Per-airfoil RMSE")
        save(fig, "worst_airfoils.png")

    if not (output / "validation_predictions.png").exists():
        sample = validation_frame.sample(n=min(5000, len(validation_frame)), random_state=42)
        fig, axes = panels()
        for name, relative_path in artifacts.items():
            model = joblib.load(output / relative_path)
            try:
                prediction = np.asarray(model.predict(sample[feature_columns])).reshape(len(sample), len(target_names))
                for index, target in enumerate(target_names):
                    if selected[target] != name:
                        continue
                    actual = sample[target].to_numpy()
                    axis = axes[0, index]
                    axis.scatter(actual, prediction[:, index], alpha=.3, s=10)
                    lower, upper = min(actual.min(), prediction[:, index].min()), max(actual.max(), prediction[:, index].max())
                    axis.plot([lower, upper], [lower, upper], "--", color="black")
                    axis.set(title=f"{target}: {name} (validation sample)", xlabel=f"Actual {target}", ylabel=f"Predicted {target}")
                    axis.grid(alpha=.3)
            finally:
                del model
                gc.collect()
        save(fig, "validation_predictions.png")
    return summary
