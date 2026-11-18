"""Fit and verify a normaliser from the complete original training partition."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from airfoil_surrogate.data.features import (
    INPUT_NAMES,
    TARGET_NAMES,
    NodeFeatures,
    Normaliser,
)
from airfoil_surrogate.data.original import load_staged_features
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import read_complete_partition


def fit_checked_normaliser(features: Iterable[NodeFeatures]) -> Normaliser:
    """Check the pooled normaliser against merged centred moments in one pass."""
    n = 0
    mean = np.zeros(len(INPUT_NAMES) + len(TARGET_NAMES), dtype=np.float64)
    squared_deviations = np.zeros_like(mean)

    def audit() -> Iterable[NodeFeatures]:
        nonlocal n, mean, squared_deviations
        for case in features:
            values = np.column_stack((case.x, case.y))
            count = len(values)
            case_mean = values.mean(axis=0)
            delta = case_mean - mean
            total = n + count
            squared_deviations += ((values - case_mean) ** 2).sum(axis=0)
            squared_deviations += delta**2 * (n * count / total)
            mean += delta * (count / total)
            n = total
            yield case

    normaliser = Normaliser.fit(audit())
    for label, actual, independent in (
        ("means", np.r_[normaliser.mean_in, normaliser.mean_out], mean),
        (
            "standard deviations",
            np.r_[normaliser.std_in, normaliser.std_out],
            np.sqrt(squared_deviations / n),
        ),
    ):
        np.testing.assert_allclose(
            actual, independent, rtol=1e-8, atol=1e-12, equal_nan=False, err_msg=label
        )
    return normaliser


def fit_staged_training(root: Path, pin: dict, out: Path) -> None:
    """Verify each training mesh and write a create-only normaliser with provenance."""
    rows, stage_sha256 = read_complete_partition(root, "train", pin)
    if out.exists():
        raise FileExistsError(out)

    def features() -> Iterable[NodeFeatures]:
        for row in rows:
            yield load_staged_features(root, row)

    normaliser = fit_checked_normaliser(features())
    with out.open("x") as stream:
        json.dump(
            normaliser.to_dict()
            | {
                "fit_stage_sha256": stage_sha256,
                "fit_cases": [row["id"] for row in rows],
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_stage", type=Path, help="complete train staging directory"
    )
    parser.add_argument("--out", type=Path, required=True, help="new JSON file")
    args = parser.parse_args()
    fit_staged_training(args.train_stage, json.loads(PIN.read_text()), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
