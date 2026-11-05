"""Keep real estimator-fitting tests on explicitly enabled Kaggle runtimes."""

import os
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    gpu = item.get_closest_marker("gpu") is not None
    if not gpu and item.get_closest_marker("cloud_fit") is None:
        return
    enabled = os.environ.get("KAGGLE_GPU_TESTS") == "1"
    if not gpu:
        enabled = enabled or os.environ.get("KAGGLE_CPU_TESTS") == "1"
    if not enabled or not Path("/kaggle/working").is_dir():
        pytest.skip("Real model fitting requires an explicitly enabled Kaggle runtime.")
