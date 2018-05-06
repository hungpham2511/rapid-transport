import pytest
import numpy as np
from toppra_app.utils import compute_area_ndim


def test_2d():
    cases = (
        ([[0, 1], [0, 2]], 1),
        ([[0, 1], [1, 0]], np.sqrt(2))
    )

    for input_, outputs_ in cases:
        np.testing.assert_allclose(compute_area_ndim(input_), outputs_)


def test_3d():
    cases = (
        ([[1, 0, 0], [0, 1, 0], [0, 0, 0]], 0.5),
        ([[1, 0, 0], [0, 0, 1], [0, 0, 0]], 0.5),
    )

    for input_, outputs_ in cases:
        np.testing.assert_allclose(compute_area_ndim(input_), outputs_)

