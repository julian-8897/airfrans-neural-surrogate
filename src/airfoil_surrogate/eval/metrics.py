"""Per-case prototype metrics and equal-case summaries."""

import numpy as np

from airfoil_surrogate.data.features import Normaliser


def relative_l2(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Channel-wise ratios, with NaN for a zero reference norm."""
    numerator = np.linalg.norm(error, axis=0)
    denominator = np.linalg.norm(reference, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )


def field_errors(
    prediction: np.ndarray,
    reference: np.ndarray,
    surf: np.ndarray,
    normaliser: Normaliser,
) -> dict[str, np.ndarray]:
    """Take normalised predictions and raw targets in matching node order."""
    error = prediction - normaliser.targets(reference)
    physical_error = error * (normaliser.std_out + normaliser.eps)
    result = {}
    for region, mask in (("volume", ~surf), ("surface", surf)):
        result[f"{region}_mse"] = np.mean(error[mask] ** 2, axis=0)
        result[f"{region}_relative_l2"] = relative_l2(
            physical_error[mask], reference[mask]
        )
    return result


def summarise(values: np.ndarray) -> dict[str, list[float | int | None]]:
    """Summarise per-case values by channel, counting defined cases."""
    summary = {"mean": [], "p95": [], "p99": [], "n_defined": []}
    for channel in values.T:
        defined = channel[np.isfinite(channel)]
        p95, p99 = np.quantile(defined, [0.95, 0.99]) if len(defined) else (None, None)
        summary["mean"].append(float(defined.mean()) if len(defined) else None)
        summary["p95"].append(float(p95) if p95 is not None else None)
        summary["p99"].append(float(p99) if p99 is not None else None)
        summary["n_defined"].append(len(defined))
    return summary
