"""GPU-trained MLP with torch-free, portable NumPy inference.

Targets are already in fitting space (normally Cl, log(Cd), Cm). The caller
must supply a frozen, geometry-disjoint validation split, never the test set.
PyTorch is imported only by fit; saved estimators contain no torch objects.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data.features import FLOW_COLUMNS
from src.geometry.descriptors import DESCRIPTOR_NAMES

RAW_FEATURE_NAMES = (
    *(f"thickness_{i:03d}" for i in range(50)),
    *(f"camber_{i:03d}" for i in range(50)),
    *DESCRIPTOR_NAMES, *FLOW_COLUMNS, "Ncrit",
)
EXTRA_FEATURE_NAMES = ("alpha_sin2", "alpha_sin_sq")


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """SiLU/RAdam regressor; predict returns fitting-space, not physical Cd.

    ``device='cuda'`` requires CUDA and never falls back. ``auto`` explicitly
    warns when using CPU. Run fitting/experiments on Kaggle, not locally.
    Huber delta is measured in standardized fitting-target units. Target
    weights are experiment choices, not numeric weights quoted from a paper.
    With symmetry enabled, target_names defines the order and known parity;
    array targets must follow that order and DataFrames must match it exactly.
    Optional Cdp supervision is identity-scaled, missing-label aware, and never
    changes the primary loss denominator or validation stopping criterion.
    """

    def __init__(
        self,
        hidden_layers=(128, 128, 64),
        max_iter=300,
        batch_size=4096,
        learning_rate_init=0.001,
        weight_decay=1e-5,
        loss="huber",
        huber_delta=0.05,
        target_weights=(1.0, 2.0, 0.5),
        symmetry=True,
        target_names=("Cl", "Cd", "Cm"),
        patience=30,
        min_delta=1e-6,
        lr_patience=10,
        lr_factor=0.5,
        random_state=42,
        device="auto",
        auxiliary_weight=0.0,
    ):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.weight_decay = weight_decay
        self.loss = loss
        self.huber_delta = huber_delta
        self.target_weights = target_weights
        self.symmetry = symmetry
        self.target_names = target_names
        self.patience = patience
        self.min_delta = min_delta
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.random_state = random_state
        self.device = device
        self.auxiliary_weight = auxiliary_weight

    def _validate_params(self):
        for name in ("max_iter", "batch_size", "patience", "lr_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (not isinstance(self.hidden_layers, (tuple, list)) or not self.hidden_layers
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1
                       for n in self.hidden_layers)):
            raise ValueError("hidden_layers must contain positive integer widths.")
        for name in ("learning_rate_init", "weight_decay", "huber_delta", "min_delta", "lr_factor", "auxiliary_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.learning_rate_init <= 0 or self.huber_delta <= 0:
            raise ValueError("learning_rate_init and huber_delta must be positive.")
        if self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("weight_decay and min_delta must be nonnegative.")
        if self.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be nonnegative.")
        if not 0 < self.lr_factor < 1:
            raise ValueError("lr_factor must be between zero and one.")
        if self.loss not in ("mse", "huber"):
            raise ValueError("loss must be 'mse' or 'huber'.")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'.")
        if not isinstance(self.symmetry, (bool, np.bool_)):
            raise ValueError("symmetry must be boolean.")
        if (not isinstance(self.target_names, tuple) or not self.target_names
                or any(not isinstance(name, str) for name in self.target_names)
                or len(set(self.target_names)) != len(self.target_names)):
            raise ValueError("target_names must be a nonempty tuple of unique target names.")
        if not set(self.target_names).issubset({"Cl", "Cd", "Cm"}):
            raise ValueError("target_names must contain only primary targets: Cl, Cd (logCd), and Cm.")
        if self.auxiliary_weight > 0 and self.target_names != ("Cl", "Cd", "Cm"):
            raise ValueError("Auxiliary Cdp requires target_names=('Cl', 'Cd', 'Cm') in that order.")
        if (not isinstance(self.target_weights, (tuple, list))
                or len(self.target_weights) != len(self.target_names)
                or any(isinstance(w, bool) or not isinstance(w, Real)
                       or not np.isfinite(w) or w <= 0 for w in self.target_weights)):
            raise ValueError("target_weights must contain one finite positive weight per target.")
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state must be an integer seed in [0, 2**32).")

    @staticmethod
    def _feature_array(X, expected_names=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a raw v2 DataFrame with the standard 113 named features.")
        names = tuple(X.columns)
        if len(names) != len(RAW_FEATURE_NAMES) or set(names) != set(RAW_FEATURE_NAMES):
            missing = sorted(set(RAW_FEATURE_NAMES).difference(names))
            extra = sorted(set(names).difference(RAW_FEATURE_NAMES), key=str)
            raise ValueError(f"Expected standard 113-feature v2 schema; missing={missing}, extra={extra}.")
        if expected_names is not None and names != tuple(expected_names):
            raise ValueError("X feature names and order must match the training DataFrame exactly.")
        if X.empty:
            raise ValueError("X must contain at least one row.")
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
               for dtype in X.dtypes):
            raise ValueError("X features must be real numeric values.")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                values = X.to_numpy(dtype=np.float32, copy=True, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError("X features must be finite real numeric float32 values.") from exc
        if not np.isfinite(values).all():
            raise ValueError("X features must be finite and representable as float32.")
        radians = np.deg2rad(values[:, names.index("Alpha")])
        return np.column_stack((values, np.sin(2 * radians), np.sin(radians) ** 2))

    def _target_array(self, y, n_rows):
        if isinstance(y, pd.DataFrame) and tuple(y.columns) != self.target_names:
            raise ValueError("y columns must match target_names order; Cd must already be log-transformed.")
        try:
            values = np.asarray(y)
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError("y must contain real numeric fitting-space targets.")
            with np.errstate(over="ignore", invalid="ignore"):
                values = values.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must contain finite numeric fitting-space targets.") from exc
        if values.shape != (n_rows, len(self.target_names)):
            raise ValueError(f"y must have shape ({n_rows}, {len(self.target_names)}) in target_names order.")
        if not np.isfinite(values).all():
            raise ValueError("y must contain finite float32 fitting-space targets.")
        return values

    @staticmethod
    def _reflection_signs(names):
        odd = {"max_camber", "Alpha", "alpha_sin", "alpha_sin2"}
        return np.array([
            -1.0 if name in odd or (name.startswith("camber_") and name[7:].isdigit()) else 1.0
            for name in (*names, *EXTRA_FEATURE_NAMES)
        ], dtype=np.float32)

    @staticmethod
    def _auxiliary_array(values, n_rows):
        """Cdp is signed, identity-space supervision; NaN denotes missingness."""
        if values is None:
            return np.full((n_rows, 1), np.nan, dtype=np.float32)
        if isinstance(values, pd.DataFrame) and tuple(values.columns) != ("Cdp",):
            raise ValueError("auxiliary_targets DataFrame must contain only Cdp.")
        try:
            if isinstance(values, (tuple, list)) and any(
                isinstance(value, (bool, np.bool_)) for value in np.asarray(values, dtype=object).flat
            ):
                raise ValueError("Boolean Cdp labels are invalid, including mixed numeric lists.")
            values = np.asarray(values)
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError("auxiliary_targets must be real numeric Cdp, not bool.")
            with np.errstate(over="ignore", invalid="ignore"):
                values = values.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("auxiliary_targets must be real numeric float32 Cdp, with NaN for missing labels.") from exc
        if values.shape == (n_rows,):
            values = values[:, None]
        if values.shape != (n_rows, 1):
            raise ValueError(f"auxiliary_targets must have shape ({n_rows},) or ({n_rows}, 1).")
        if np.isinf(values).any():
            raise ValueError("auxiliary_targets cannot contain Inf or exceed the float32 range.")
        return values

    def fit(self, X, y, *, validation_data=None, auxiliary_targets=None,
            validation_auxiliary_targets=None):
        """Fit using primary validation loss only for scheduling/stopping.

        Cdp labels are separate from X/y, shape (n,) or (n, 1), with NaN for
        missing labels. Enabled supervision requires observed training labels;
        validation Cdp may be absent. Weight zero rejects supplied Cdp labels.
        """
        self._validate_params()
        if (not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2
                or any(value is None for value in validation_data)):
            raise ValueError("validation_data=(X_val, y_val) is required; supply frozen geometry-disjoint validation, not test.")
        x_train = self._feature_array(X)
        y_train = self._target_array(y, len(x_train))
        x_val = self._feature_array(validation_data[0], X.columns)
        y_val = self._target_array(validation_data[1], len(x_val))
        auxiliary_enabled = self.auxiliary_weight > 0
        if not auxiliary_enabled and (auxiliary_targets is not None or validation_auxiliary_targets is not None):
            raise ValueError("auxiliary_targets require auxiliary_weight > 0; omit both auxiliary arrays for the baseline.")
        if auxiliary_enabled:
            auxiliary_train = self._auxiliary_array(auxiliary_targets, len(x_train))
            auxiliary_val = self._auxiliary_array(validation_auxiliary_targets, len(x_val))
            if not np.isfinite(auxiliary_train).any():
                raise ValueError("Auxiliary Cdp requires at least one observed training label.")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Fitting requires PyTorch in the Kaggle training environment; NumPy predict does not.") from exc
        cuda_available = torch.cuda.is_available()
        if self.device == "cuda" and not cuda_available:
            raise RuntimeError("device='cuda' requires an available CUDA GPU; enable a Kaggle GPU accelerator.")
        accelerator = "cuda" if self.device != "cpu" and cuda_available else "cpu"
        if self.device == "auto" and accelerator == "cpu":
            warnings.warn("device='auto': CUDA unavailable, explicitly falling back to CPU.", RuntimeWarning)
        device = torch.device(accelerator)
        torch.manual_seed(int(self.random_state))
        if accelerator == "cuda":
            torch.cuda.manual_seed_all(int(self.random_state))
        generator = torch.Generator(device="cpu").manual_seed(int(self.random_state))

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.n_outputs_ = len(self.target_names)
        self.symmetry_ = bool(self.symmetry)
        self.reflection_signs_ = self._reflection_signs(X.columns)
        self.target_signs_ = np.array([
            -1.0 if name in ("Cl", "Cm") else 1.0 for name in self.target_names
        ], dtype=np.float32)
        self.input_scaler_ = StandardScaler().fit(x_train)
        if self.symmetry_:
            # Train-only reflection, in bounded CPU chunks; no validation statistics.
            for start in range(0, len(x_train), self.batch_size):
                self.input_scaler_.partial_fit(x_train[start:start + self.batch_size] * self.reflection_signs_)
        self.target_scaler_ = StandardScaler().fit(y_train)
        self.__dict__.pop("auxiliary_scaler_", None)
        if auxiliary_enabled:
            self.auxiliary_scaler_ = StandardScaler().fit(auxiliary_train[np.isfinite(auxiliary_train)].reshape(-1, 1))
        input_mean = torch.as_tensor(self.input_scaler_.mean_, dtype=torch.float32, device=device)
        input_scale = torch.as_tensor(self.input_scaler_.scale_, dtype=torch.float32, device=device)
        target_mean = torch.as_tensor(self.target_scaler_.mean_, dtype=torch.float32, device=device)
        target_scale = torch.as_tensor(self.target_scaler_.scale_, dtype=torch.float32, device=device)
        reflection = torch.as_tensor(self.reflection_signs_, device=device)
        target_signs = torch.as_tensor(self.target_signs_, device=device)
        network_mean, network_scale, network_signs = target_mean, target_scale, target_signs
        if auxiliary_enabled:
            auxiliary_mean = torch.as_tensor(self.auxiliary_scaler_.mean_, dtype=torch.float32, device=device)
            auxiliary_scale = torch.as_tensor(self.auxiliary_scaler_.scale_, dtype=torch.float32, device=device)
            network_mean = torch.cat((target_mean, auxiliary_mean))
            network_scale = torch.cat((target_scale, auxiliary_scale))
            network_signs = torch.cat((target_signs, torch.ones(1, device=device)))
        weights = torch.tensor(self.target_weights, dtype=torch.float32, device=device)
        widths = (x_train.shape[1], *self.hidden_layers, self.n_outputs_)
        layers = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(int(left), int(right)))
            if index < len(widths) - 2:
                layers.append(torch.nn.SiLU())
        if auxiliary_enabled:
            # Draw the unchanged primary head FIRST: Linear(width, 4) directly
            # would consume an extra weight row before drawing primary biases.
            primary = layers[-1]
            auxiliary = torch.nn.Linear(int(widths[-2]), 1)
            joint = torch.nn.Linear(int(widths[-2]), self.n_outputs_ + 1)
            with torch.no_grad():
                joint.weight.copy_(torch.cat((primary.weight, auxiliary.weight)))
                joint.bias.copy_(torch.cat((primary.bias, auxiliary.bias)))
            layers[-1] = joint
        network = torch.nn.Sequential(*layers).to(device)
        optimizer = torch.optim.RAdam(network.parameters(), lr=self.learning_rate_init, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience,
            threshold=self.min_delta, threshold_mode="abs",
        )

        def normalized_prediction(raw):
            prediction = network((raw - input_mean) / input_scale)
            if self.symmetry_:
                reflected = network((raw * reflection - input_mean) / input_scale)
                # Odd targets can have nonzero training means. Project in unscaled
                # Cl/logCd/Cm coordinates, THEN standardize for the loss.
                prediction = (prediction * network_scale + network_mean
                              + network_signs * (reflected * network_scale + network_mean)) * 0.5
                prediction = (prediction - network_mean) / network_scale
            return prediction

        def batch_loss(prediction, target):
            if self.loss == "mse":
                elementwise = (prediction - target).square()
            else:
                elementwise = torch.nn.functional.huber_loss(
                    prediction, target, reduction="none", delta=self.huber_delta,
                )
            return (elementwise * weights).sum(dim=1).mean() / weights.sum()

        def auxiliary_loss(prediction, target):
            mask = ~torch.isnan(target)
            count = int(mask.sum().item())
            # Select before normalization/loss: multiplying NaN losses by zero
            # would still poison gradients. Empty selection gives zero gradient.
            target = (target[mask] - auxiliary_mean) / auxiliary_scale
            prediction = prediction[mask]
            if self.loss == "mse":
                elementwise = (prediction - target).square()
            else:
                elementwise = torch.nn.functional.huber_loss(
                    prediction, target, reduction="none", delta=self.huber_delta,
                )
            return elementwise.sum() / max(count, 1), count

        # Entire datasets stay float32 on CPU; only bounded batches go to GPU.
        train_x = torch.from_numpy(x_train)
        train_y = torch.from_numpy(y_train)
        val_x = torch.from_numpy(x_val)
        val_y = torch.from_numpy(y_val)
        if auxiliary_enabled:
            train_auxiliary = torch.from_numpy(auxiliary_train)
            val_auxiliary = torch.from_numpy(auxiliary_val)
        best_loss = float("inf")
        patience_loss = float("inf")
        best_state = None
        best_auxiliary_loss = None
        stale_epochs = 0
        history = []
        stop_reason = "max_iter"
        for epoch in range(1, self.max_iter + 1):
            network.train()
            order = torch.randperm(len(train_x), generator=generator)
            train_sum = 0.0
            train_total_sum = 0.0
            train_auxiliary_sum, train_auxiliary_count = 0.0, 0
            for start in range(0, len(train_x), self.batch_size):
                indices = order[start:start + self.batch_size]
                raw = train_x[indices].to(device)
                target = (train_y[indices].to(device) - target_mean) / target_scale
                optimizer.zero_grad(set_to_none=True)
                prediction = normalized_prediction(raw)
                primary_value = batch_loss(prediction[:, :self.n_outputs_], target)
                value = primary_value
                if auxiliary_enabled:
                    auxiliary_value, count = auxiliary_loss(
                        prediction[:, self.n_outputs_:], train_auxiliary[indices].to(device),
                    )
                    value = primary_value + self.auxiliary_weight * auxiliary_value
                    train_auxiliary_sum += float(auxiliary_value.detach().item()) * count
                    train_auxiliary_count += count
                if not torch.isfinite(value):
                    raise RuntimeError("Nonfinite training loss; check target scales and learning_rate_init.")
                value.backward()
                optimizer.step()
                train_sum += float(primary_value.detach().item()) * len(indices)
                train_total_sum += float(value.detach().item()) * len(indices)
            network.eval()
            val_sum = 0.0
            val_auxiliary_sum, val_auxiliary_count = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(val_x), self.batch_size):
                    raw = val_x[start:start + self.batch_size].to(device)
                    target = (val_y[start:start + self.batch_size].to(device) - target_mean) / target_scale
                    prediction = normalized_prediction(raw)
                    value = batch_loss(prediction[:, :self.n_outputs_], target)
                    val_sum += float(value.item()) * len(raw)
                    if auxiliary_enabled:
                        auxiliary_value, count = auxiliary_loss(
                            prediction[:, self.n_outputs_:], val_auxiliary[start:start + self.batch_size].to(device),
                        )
                        val_auxiliary_sum += float(auxiliary_value.item()) * count
                        val_auxiliary_count += count
            validation_loss = val_sum / len(val_x)
            train_auxiliary_loss = train_auxiliary_sum / train_auxiliary_count if train_auxiliary_count else None
            validation_auxiliary_loss = val_auxiliary_sum / val_auxiliary_count if val_auxiliary_count else None
            if not np.isfinite(validation_loss):
                raise RuntimeError("Nonfinite validation loss; check validation values and target scales.")
            if validation_auxiliary_loss is not None and not np.isfinite(validation_auxiliary_loss):
                raise RuntimeError("Nonfinite auxiliary validation loss; check Cdp values and target scales.")
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_auxiliary_loss = validation_auxiliary_loss
                self.best_epoch_ = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
            if validation_loss < patience_loss - self.min_delta:
                patience_loss = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append({"epoch": epoch, "train_loss": train_sum / len(train_x),
                            "validation_loss": validation_loss, "learning_rate": learning_rate,
                            "train_primary_loss": train_sum / len(train_x),
                            "validation_primary_loss": validation_loss,
                            "train_auxiliary_loss": train_auxiliary_loss,
                            "validation_auxiliary_loss": validation_auxiliary_loss,
                            "train_total_loss": train_total_sum / len(train_x)})
            if len(history) > 300:
                history.pop(0)
            if epoch == 1 or epoch % 10 == 0:
                print(f"TorchMLP epoch {epoch}/{self.max_iter}: train={train_sum / len(train_x):.7g} "
                      f"validation={validation_loss:.7g} best={best_loss:.7g} lr={learning_rate:.3g}", flush=True)
            if stale_epochs >= self.patience:
                stop_reason = "validation_patience"
                break

        network.load_state_dict(best_state)
        self.coefs_ = [layer.weight.detach().cpu().numpy().copy() for layer in network if isinstance(layer, torch.nn.Linear)]
        self.intercepts_ = [layer.bias.detach().cpu().numpy().copy() for layer in network if isinstance(layer, torch.nn.Linear)]
        self.n_iter_ = epoch
        self.best_loss_ = self.loss_ = best_loss
        cuda_index = torch.cuda.current_device() if accelerator == "cuda" else None
        self.training_metadata_ = {
            "accelerator": accelerator, "device": str(device),
            "device_name": torch.cuda.get_device_name(cuda_index) if cuda_index is not None else "CPU",
            "cuda_device_index": cuda_index, "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "cuda_available": bool(cuda_available),
            "seed": int(self.random_state), "n_iter": epoch, "max_iter": int(self.max_iter),
            "best_epoch": self.best_epoch_, "best_loss": best_loss,
            "stop_reason": stop_reason, "stopped_early": epoch < self.max_iter,
            "patience": int(self.patience), "stale_epochs": stale_epochs, "min_delta": float(self.min_delta),
            "restored_best_checkpoint": True, "loss": self.loss, "huber_delta": float(self.huber_delta),
            "target_weights": [float(w) for w in self.target_weights], "target_names": list(self.target_names),
            "stopping_metric": "primary_normalized_validation_loss",
            "auxiliary_weight": float(self.auxiliary_weight),
            "auxiliary_name": "Cdp" if auxiliary_enabled else None,
            "auxiliary_transform": "identity" if auxiliary_enabled else None,
            "auxiliary_label_coverage": None, "auxiliary_scaler": None,
            "best_auxiliary_validation_loss": best_auxiliary_loss,
            "loss_formula": "primary weighted mean + auxiliary_weight * observed auxiliary mean",
            "target_space": "caller-supplied fitting space; Cd is already logCd",
            "loss_units": "standardized fitting targets", "symmetry": self.symmetry_,
            "learning_rate_init": float(self.learning_rate_init), "weight_decay": float(self.weight_decay),
            "optimizer": "RAdam", "weight_decay_type": "l2", "activation": "SiLU",
            "lr_patience": int(self.lr_patience), "lr_factor": float(self.lr_factor),
            "batch_size": int(self.batch_size), "hidden_layers": [int(n) for n in self.hidden_layers],
            "n_train": len(x_train), "n_validation": len(x_val),
            "n_raw_features": self.n_features_in_, "n_network_features": x_train.shape[1],
            "history": history, "history_truncated": epoch > len(history),
        }
        if auxiliary_enabled:
            coverage = {}
            for name, values in (("train", auxiliary_train), ("validation", auxiliary_val)):
                observed = values[np.isfinite(values)].astype(np.float64)
                coverage[name] = {
                    "n_rows": len(values), "n_observed": len(observed), "n_missing": len(values) - len(observed),
                    "fraction_observed": len(observed) / len(values),
                    "min": float(observed.min()) if len(observed) else None,
                    "max": float(observed.max()) if len(observed) else None,
                    "mean": float(observed.mean()) if len(observed) else None,
                }
            self.training_metadata_["auxiliary_label_coverage"] = coverage
            self.training_metadata_["auxiliary_scaler"] = {
                "mean": float(self.auxiliary_scaler_.mean_[0]), "scale": float(self.auxiliary_scaler_.scale_[0]),
                "var": float(self.auxiliary_scaler_.var_[0]), "n_samples_seen": int(self.auxiliary_scaler_.n_samples_seen_),
            }
        print(f"TorchMLP stopped: {stop_reason}; epochs={epoch}, best_epoch={self.best_epoch_}, "
              f"best_validation_loss={best_loss:.7g}; restored best checkpoint; device={device}", flush=True)
        return self

    def _latent_prediction(self, raw):
        values = ((raw - self.input_scaler_.mean_.astype(np.float32))
                  / self.input_scaler_.scale_.astype(np.float32))
        for index, (weight, bias) in enumerate(zip(self.coefs_, self.intercepts_)):
            values = values @ weight.T + bias
            if index < len(self.coefs_) - 1:
                # Stable beta=1 SiLU, including large negative extrapolations.
                values = values * np.exp(-np.logaddexp(np.float32(0), -values))
        return values

    def _numpy_forward(self, raw):
        values = self._latent_prediction(raw)[:, :self.n_outputs_]
        return values * self.target_scaler_.scale_.astype(np.float32) + self.target_scaler_.mean_.astype(np.float32)

    def predict(self, X):
        """Batched NumPy prediction; joblib loading/prediction needs no PyTorch."""
        check_is_fitted(self, ("coefs_", "intercepts_", "input_scaler_", "target_scaler_", "symmetry_"))
        raw = self._feature_array(X, self.feature_names_in_)
        prediction = np.empty((len(raw), self.n_outputs_), dtype=np.float32)
        for start in range(0, len(raw), self.batch_size):
            batch = raw[start:start + self.batch_size]
            values = self._numpy_forward(batch)
            if self.symmetry_:
                values = 0.5 * (values + self.target_signs_ * self._numpy_forward(batch * self.reflection_signs_))
            prediction[start:start + len(batch)] = values
        if not np.isfinite(prediction).all():
            raise ValueError("Predictions are nonfinite; inputs exceed the model's float32 numerical range.")
        return prediction

    def predict_auxiliary(self, X):
        """Return identity-space Cdp with shape (n, 1), using portable weights."""
        check_is_fitted(self, ("coefs_", "intercepts_", "input_scaler_", "auxiliary_scaler_", "symmetry_"))
        raw = self._feature_array(X, self.feature_names_in_)
        prediction = np.empty((len(raw), 1), dtype=np.float32)
        for start in range(0, len(raw), self.batch_size):
            batch = raw[start:start + self.batch_size]
            values = self._latent_prediction(batch)[:, self.n_outputs_:]
            if self.symmetry_:
                # Cdp is even: averaging normalized outputs preserves its offset.
                values = 0.5 * (values + self._latent_prediction(batch * self.reflection_signs_)[:, self.n_outputs_:])
            prediction[start:start + len(batch)] = (
                values * self.auxiliary_scaler_.scale_.astype(np.float32) + self.auxiliary_scaler_.mean_.astype(np.float32)
            )
        if not np.isfinite(prediction).all():
            raise ValueError("Auxiliary predictions are nonfinite; inputs exceed the model's float32 numerical range.")
        return prediction


def build_model(**kwargs):
    """Build an estimator without importing PyTorch or wrapping target transforms."""
    return TorchMLPRegressor(**kwargs)
