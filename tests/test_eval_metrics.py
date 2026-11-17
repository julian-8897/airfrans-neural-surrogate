import numpy as np
from numpy.testing import assert_allclose

from airfoil_surrogate.data.features import Normaliser
from airfoil_surrogate.eval.metrics import field_errors, summarise


def test_normalised_mse_and_physical_relative_l2():
    norm = Normaliser(np.zeros(7), np.ones(7), np.ones(4), np.full(4, 2.0), eps=0)
    truth = np.array([[3, 1, 0, 2], [1, 1, 0, 2], [1, 1, 0, 2]], dtype=float)
    prediction = norm.targets(truth)
    prediction[0] += 1
    prediction[2] -= 2
    surf = np.array([False, False, True])

    errors = field_errors(prediction, truth, surf, norm)

    assert_allclose(errors["volume_mse"], 0.5)
    assert_allclose(errors["surface_mse"], 4)
    assert_allclose(
        errors["volume_relative_l2"],
        [2 / np.sqrt(10), np.sqrt(2), np.nan, 1 / np.sqrt(2)],
    )
    assert_allclose(errors["surface_relative_l2"], [4, 4, np.nan, 2])


def test_case_quantiles_count_defined_values():
    summary = summarise(np.array([[0, np.nan], [100, np.nan]]))

    assert summary == {
        "mean": [50, None],
        "p95": [95, None],
        "p99": [99, None],
        "n_defined": [2, 0],
    }
