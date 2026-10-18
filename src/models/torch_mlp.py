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

    def _validate_params(self):
        for name in ("max_iter", "batch_size", "patience", "lr_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (not isinstance(self.hidden_layers, (tuple, list)) or not self.hidden_layers
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1
                       for n in self.hidden_layers)):
            raise ValueError("hidden_layers must contain positive integer widths.")
        for name in ("learning_rate_init", "weight_decay", "huber_delta", "min_delta", "lr_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.learning_rate_init <= 0 or self.huber_delta <= 0:
            raise ValueError("learning_rate_init and huber_delta must be positive.")
        if self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("weight_decay and min_delta must be nonnegative.")
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
        if self.symmetry and not set(self.target_names).issubset({"Cl", "Cd", "Cm"}):
            raise ValueError("Symmetry requires known target_names: Cl, Cd (logCd), and Cm.")
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

    def fit(self, X, y, *, validation_data=None):
        """Fit with explicit validation-only scheduling/stopping; no test access."""
        self._validate_params()
        if (not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2
                or any(value is None for value in validation_data)):
            raise ValueError("validation_data=(X_val, y_val) is required; supply frozen geometry-disjoint validation, not test.")
        x_train = self._feature_array(X)
        y_train = self._target_array(y, len(x_train))
        x_val = self._feature_array(validation_data[0], X.columns)
        y_val = self._target_array(validation_data[1], len(x_val))
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
        input_mean = torch.as_tensor(self.input_scaler_.mean_, dtype=torch.float32, device=device)
        input_scale = torch.as_tensor(self.input_scaler_.scale_, dtype=torch.float32, device=device)
        target_mean = torch.as_tensor(self.target_scaler_.mean_, dtype=torch.float32, device=device)
        target_scale = torch.as_tensor(self.target_scaler_.scale_, dtype=torch.float32, device=device)
        reflection = torch.as_tensor(self.reflection_signs_, device=device)
        target_signs = torch.as_tensor(self.target_signs_, device=device)
        weights = torch.tensor(self.target_weights, dtype=torch.float32, device=device)
        widths = (x_train.shape[1], *self.hidden_layers, self.n_outputs_)
        layers = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(int(left), int(right)))
            if index < len(widths) - 2:
                layers.append(torch.nn.SiLU())
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
                prediction = (prediction * target_scale + target_mean
                              + target_signs * (reflected * target_scale + target_mean)) * 0.5
                prediction = (prediction - target_mean) / target_scale
            return prediction

        def batch_loss(prediction, target):
            if self.loss == "mse":
                elementwise = (prediction - target).square()
            else:
                elementwise = torch.nn.functional.huber_loss(
                    prediction, target, reduction="none", delta=self.huber_delta,
                )
            return (elementwise * weights).sum(dim=1).mean() / weights.sum()

        # Entire datasets stay float32 on CPU; only bounded batches go to GPU.
        train_x = torch.from_numpy(x_train)
        train_y = torch.from_numpy(y_train)
        val_x = torch.from_numpy(x_val)
        val_y = torch.from_numpy(y_val)
        best_loss = float("inf")
        patience_loss = float("inf")
        best_state = None
        stale_epochs = 0
        history = []
        stop_reason = "max_iter"
        for epoch in range(1, self.max_iter + 1):
            network.train()
            order = torch.randperm(len(train_x), generator=generator)
            train_sum = 0.0
            for start in range(0, len(train_x), self.batch_size):
                indices = order[start:start + self.batch_size]
                raw = train_x[indices].to(device)
                target = (train_y[indices].to(device) - target_mean) / target_scale
                optimizer.zero_grad(set_to_none=True)
                value = batch_loss(normalized_prediction(raw), target)
                if not torch.isfinite(value):
                    raise RuntimeError("Nonfinite training loss; check target scales and learning_rate_init.")
                value.backward()
                optimizer.step()
                train_sum += float(value.detach().item()) * len(indices)
            network.eval()
            val_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_x), self.batch_size):
                    raw = val_x[start:start + self.batch_size].to(device)
                    target = (val_y[start:start + self.batch_size].to(device) - target_mean) / target_scale
                    value = batch_loss(normalized_prediction(raw), target)
                    val_sum += float(value.item()) * len(raw)
            validation_loss = val_sum / len(val_x)
            if not np.isfinite(validation_loss):
                raise RuntimeError("Nonfinite validation loss; check validation values and target scales.")
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                self.best_epoch_ = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
            if validation_loss < patience_loss - self.min_delta:
                patience_loss = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append({"epoch": epoch, "train_loss": train_sum / len(train_x),
                            "validation_loss": validation_loss, "learning_rate": learning_rate})
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
        print(f"TorchMLP stopped: {stop_reason}; epochs={epoch}, best_epoch={self.best_epoch_}, "
              f"best_validation_loss={best_loss:.7g}; restored best checkpoint; device={device}", flush=True)
        return self

    def _numpy_forward(self, raw):
        values = ((raw - self.input_scaler_.mean_.astype(np.float32))
                  / self.input_scaler_.scale_.astype(np.float32))
        for index, (weight, bias) in enumerate(zip(self.coefs_, self.intercepts_)):
            values = values @ weight.T + bias
            if index < len(self.coefs_) - 1:
                # Stable beta=1 SiLU, including large negative extrapolations.
                values = values * np.exp(-np.logaddexp(np.float32(0), -values))
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


def build_model(**kwargs):
    """Build an estimator without importing PyTorch or wrapping target transforms."""
    return TorchMLPRegressor(**kwargs)
