"""Weighted native-mesh errors and shape-level transfer statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from airfoil_surrogate.data.features import TARGET_NAMES
from airfoil_surrogate.data.original import OriginalMesh
from airfoil_surrogate.eval.forces import coefficient_spearman
from airfoil_surrogate.eval.metrics import summarise
from airfoil_surrogate.eval.original import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED

GROUPS = ("uiuc", "naca")
COEFFICIENTS = ("CL", "CD")
METRICS = (
    "volume_mse",
    "surface_mse",
    "volume_relative_l2",
    "surface_relative_l2",
    "volume_area_weighted_mse",
    "surface_arc_length_weighted_mse",
    "coefficient_absolute_error",
    "coefficient_relative_error",
    "wall_pressure_rmse",
    "wall_shear_rmse",
)


def weighted_mse(mesh: OriginalMesh, error: np.ndarray) -> dict[str, np.ndarray]:
    """Distribute cell areas and segment lengths equally to incident nodes."""
    grid, wall = mesh.internal, mesh.aerofoil
    sizes = np.diff(grid.cell_offsets)
    area = np.asarray(
        grid.compute_cell_sizes(length=False, volume=False)["Area"], dtype=np.float64
    )
    if np.any(sizes < 3) or not np.all(np.isfinite(area) & (area > 0)):
        raise ValueError("transfer volume must have positive-area planar cells")
    weights = np.bincount(
        grid.cell_connectivity,
        weights=np.repeat(area / sizes, sizes),
        minlength=grid.n_points,
    )
    off_wall = np.ones(grid.n_points, dtype=bool)
    off_wall[mesh.wall_nodes] = False
    off_wall_weights = weights[off_wall]

    lines = wall.lines
    if (
        wall.n_cells != wall.n_lines
        or lines.size != 3 * wall.n_lines
        or np.any(lines[::3] != 2)
    ):
        raise ValueError("transfer wall must consist of two-point segments")
    endpoints = lines.reshape(-1, 3)[:, 1:]
    points = np.asarray(wall.points, dtype=np.float64)
    length = np.linalg.norm(points[endpoints[:, 0]] - points[endpoints[:, 1]], axis=1)
    wall_weights = np.bincount(
        endpoints.ravel(), weights=np.repeat(length / 2, 2), minlength=wall.n_points
    )
    if (
        not np.all(np.isfinite(length) & (length > 0))
        or np.any(wall_weights <= 0)
        or np.any(off_wall_weights <= 0)
    ):
        raise ValueError("transfer regions must have finite positive lumped weights")
    squared = error**2
    return {
        "volume_area_weighted_mse": np.average(
            squared[off_wall], axis=0, weights=off_wall_weights
        ),
        "surface_arc_length_weighted_mse": np.average(
            squared[mesh.wall_nodes], axis=0, weights=wall_weights
        ),
    }


def ranking(ids: list[str], prediction: np.ndarray, reference: np.ndarray) -> dict:
    """Use average descending ranks and count exact ties separately."""
    predicted_rank = rankdata(-prediction, method="average")
    reference_rank = rankdata(-reference, method="average")
    left, right = np.triu_indices(len(ids), k=1)
    pred_order = np.sign(prediction[left] - prediction[right])
    ref_order = np.sign(reference[left] - reference[right])
    discordant = pred_order * ref_order < 0
    return {
        "ranks": [
            {
                "case": case,
                "predicted_rank": float(p),
                "reference_rank": float(r),
                "predicted_minus_reference_rank": float(p - r),
            }
            for case, p, r in zip(ids, predicted_rank, reference_rank, strict=True)
        ],
        "discordant_pairs": [
            [ids[i], ids[j]]
            for i, j in zip(left[discordant], right[discordant], strict=True)
        ],
        "predicted_ties": int(np.count_nonzero(pred_order == 0)),
        "reference_ties": int(np.count_nonzero(ref_order == 0)),
        "ties_in_both": int(np.count_nonzero((pred_order == 0) & (ref_order == 0))),
    }


def _resampled_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    draws = values[indices]
    defined = np.isfinite(draws)
    counts = defined.sum(axis=1)
    total = np.where(defined, draws, 0).sum(axis=1)
    return np.divide(total, counts, out=np.full_like(total, np.nan), where=counts != 0)


def _resampled_spearman(
    prediction: np.ndarray, reference: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    draws = np.empty((len(indices), len(COEFFICIENTS)))
    for column in range(len(COEFFICIENTS)):
        first = rankdata(prediction[:, column][indices], axis=1, method="average")
        second = rankdata(reference[:, column][indices], axis=1, method="average")
        first -= first.mean(axis=1, keepdims=True)
        second -= second.mean(axis=1, keepdims=True)
        normalisation = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        draws[:, column] = np.divide(
            (first * second).sum(axis=1),
            normalisation,
            out=np.full(len(indices), np.nan),
            where=normalisation != 0,
        )
    return draws.clip(-1.0, 1.0)


def _difference(
    uiuc: list[float | None], naca: list[float | None], draws: np.ndarray
) -> list[dict]:
    result = []
    for first, second, values in zip(uiuc, naca, draws.T, strict=True):
        defined = values[np.isfinite(values)]
        result.append(
            {
                "difference": first - second
                if first is not None and second is not None
                else None,
                "interval95": np.quantile(defined, [0.025, 0.975]).tolist()
                if len(defined)
                else None,
                "n_draws": len(defined),
                "n_undefined_draws": len(values) - len(defined),
            }
        )
    return result


def summarise_transfer(rows: list[dict]) -> tuple[dict, dict]:
    """Summarise cohorts and bootstrap their difference with independent draws."""
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    groups, ranks, metrics_draws, spearman_draws = {}, {}, {}, {}
    for group in GROUPS:
        cases = [row for row in rows if row["group"] == group]
        ids = [row["case"] for row in cases]
        prediction = np.asarray([row["transolver"]["coefficients"] for row in cases])[
            :, :, 0
        ]
        raw = np.asarray([row["transolver"]["raw_coefficients"] for row in cases])[
            :, :, 0
        ]
        reference = np.asarray([row["reference_coefficients"] for row in cases])[
            :, :, 0
        ]
        indices = rng.integers(len(cases), size=(BOOTSTRAP_DRAWS, len(cases)))
        metrics, metrics_draws[group] = {}, {}
        for name in METRICS:
            data = np.asarray(
                [row["transolver"][name] for row in cases], dtype=np.float64
            )
            if data.ndim == 1:
                data = data[:, None]
            metrics[name] = summarise(data)
            metrics_draws[group][name] = _resampled_means(data, indices)
        spearman_draws[group] = _resampled_spearman(prediction, reference, indices)
        groups[group] = {
            "case_count": len(cases),
            "metrics": metrics,
            "spearman": coefficient_spearman(prediction, reference),
            "spearman_raw": coefficient_spearman(raw, reference),
            "spearman_n_draws": np.isfinite(spearman_draws[group]).sum(axis=0).tolist(),
            "spearman_n_undefined_draws": np.isnan(spearman_draws[group])
            .sum(axis=0)
            .tolist(),
        }
        ranks[group] = {
            name: ranking(ids, prediction[:, index], reference[:, index])
            for index, name in enumerate(COEFFICIENTS)
        }
    uiuc, naca = GROUPS
    return {
        "fields": TARGET_NAMES,
        "coefficients": COEFFICIENTS,
        "force_components": ("total", "pressure", "viscous"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "groups": groups,
        "uiuc_minus_naca": {
            "metrics": {
                name: _difference(
                    groups[uiuc]["metrics"][name]["mean"],
                    groups[naca]["metrics"][name]["mean"],
                    metrics_draws[uiuc][name] - metrics_draws[naca][name],
                )
                for name in METRICS
            },
            "spearman": _difference(
                groups[uiuc]["spearman"],
                groups[naca]["spearman"],
                spearman_draws[uiuc] - spearman_draws[naca],
            ),
        },
    }, ranks
