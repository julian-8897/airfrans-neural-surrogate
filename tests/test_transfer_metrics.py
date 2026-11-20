import numpy as np
import pytest
from numpy.testing import assert_allclose

pv = pytest.importorskip("pyvista")
metrics = pytest.importorskip("airfoil_surrogate.eval.transfer_metrics")
original = pytest.importorskip("airfoil_surrogate.data.original")


def test_lumped_mse_for_unequal_areas_and_lengths():
    points = np.array([(x, y, 0.0) for y in (0, 1) for x in (0, 1, 4)])
    grid = pv.UnstructuredGrid(
        [4, 0, 1, 4, 3, 4, 1, 2, 5, 4], [pv.CellType.QUAD] * 2, points
    )
    wall_nodes = np.array([2, 1, 0])
    wall = pv.PolyData(points[wall_nodes], lines=[2, 0, 1, 2, 1, 2])
    mesh = original.match_original_mesh(grid, wall)
    error = np.array([[1, 2], [2, 2], [3, 2], [3, 1], [2, 1], [1, 1]])

    result = metrics.weighted_mse(mesh, error)

    assert_allclose(result["surface_arc_length_weighted_mse"], [44 / 8, 4])
    assert_allclose(result["volume_area_weighted_mse"], [28 / 8, 1])


def test_report_discordances_and_exact_ties():
    result = metrics.ranking(
        ["a", "b", "c", "d"],
        np.array([5.0, 3.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 3.0, 4.0]),
    )
    assert result["discordant_pairs"] == [
        ["a", "b"],
        ["a", "d"],
        ["b", "c"],
        ["c", "d"],
    ]
    assert result["predicted_ties"] == 2
    assert result["reference_ties"] == 1
    assert result["ties_in_both"] == 1
    assert result["ranks"][0] == {
        "case": "a",
        "predicted_rank": 1.5,
        "reference_rank": 4.0,
        "predicted_minus_reference_rank": -2.5,
    }


def test_transfer_bootstrap_reports_differences_and_undefined_draws():
    rows = []
    for group, error, sign in (("uiuc", 3.0, -1), ("naca", 1.0, 1)):
        for index in range(3):
            pred = [[sign * index, 0, 0], [0.5, 0, 0]]
            ref = [[index, 0, 0], [1.0, 0, 0]]
            rows.append(
                {
                    "case": f"{group}-{index}",
                    "group": group,
                    "reference_coefficients": ref,
                    "transolver": {
                        **{name: [error, None] for name in metrics.METRICS},
                        "coefficients": pred,
                        "raw_coefficients": pred,
                    },
                }
            )

    summary, rankings = metrics.summarise_transfer(rows)

    delta = summary["uiuc_minus_naca"]
    mse, undefined = delta["metrics"]["volume_area_weighted_mse"]
    assert mse == {
        "difference": 2.0,
        "interval95": [2.0, 2.0],
        "n_draws": 10_000,
        "n_undefined_draws": 0,
    }
    assert undefined == {
        "difference": None,
        "interval95": None,
        "n_draws": 0,
        "n_undefined_draws": 10_000,
    }
    lift, drag = delta["spearman"]
    assert_allclose(lift["difference"], -2.0)
    assert_allclose(lift["interval95"], [-2.0, -2.0])
    assert 0 < lift["n_undefined_draws"] < 10_000
    assert lift["n_draws"] + lift["n_undefined_draws"] == 10_000
    assert drag == undefined
    assert len(rankings["uiuc"]["CL"]["discordant_pairs"]) == 3
    assert not rankings["naca"]["CL"]["discordant_pairs"]
