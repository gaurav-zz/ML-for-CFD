import numpy as np
import pytest

from src.geometry.naca import generate_naca4, parse_naca4
from src.geometry.representation import normalize_coordinates


def test_parse_naca4():
    assert parse_naca4("NACA2412") == (.02, .4, .12)


def test_invalid_naca_code():
    with pytest.raises(ValueError):
        parse_naca4("241")


def test_symmetric_airfoil():
    upper, lower = generate_naca4("0012", 32)
    assert np.allclose(upper[:, 1], -lower[:, 1])


def test_cambered_airfoil_and_normalization():
    upper, lower = generate_naca4("2412", 32)
    coordinates = normalize_coordinates(np.vstack((upper, lower)))
    assert coordinates[:, 0].min() == pytest.approx(0)
    assert coordinates[:, 0].max() == pytest.approx(1)
    assert coordinates[:, 1].max() > 0
