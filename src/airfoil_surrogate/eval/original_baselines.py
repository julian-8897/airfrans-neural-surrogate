"""Fit regional field means and NACA coefficient baselines on original training data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from airfoil_surrogate.data.features import TARGET_NAMES
from airfoil_surrogate.data.original import load_original_mesh
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import (
    read_complete_partition,
    verified_mesh_paths,
)
from airfoil_surrogate.eval.forces import (
    AIRFRANS_VISCOSITY,
    OriginalForceSimulation,
    force_coefficients,
)


def naca_design(case_id: str) -> tuple[str, np.ndarray]:
    """Return the NACA family and its parameters followed by Reynolds and AoA."""
    prefix, turbulence, inlet, angle, *shape = case_id.split("_")
    if (prefix, turbulence) != ("airFoil2D", "SST") or len(shape) not in (3, 4):
        raise ValueError(f"invalid original NACA case ID {case_id!r}")
    speed, angle_deg = float(inlet), float(angle)
    inputs = np.array([*map(float, shape), speed / AIRFRANS_VISCOSITY, angle_deg])
    if speed <= 0 or not np.isfinite(inputs).all():
        raise ValueError(f"invalid operating conditions in {case_id!r}")
    return f"naca{len(shape) + 1}", inputs


def _quadratic_terms(z: np.ndarray) -> np.ndarray:
    left, right = np.triu_indices(z.shape[1])
    return np.column_stack((z, z[:, left] * z[:, right]))


@dataclass(frozen=True)
class CoefficientRidge:
    input_mask: np.ndarray
    centre: np.ndarray
    scale: np.ndarray
    term_mask: np.ndarray
    intercept: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, inputs: np.ndarray, targets: np.ndarray) -> CoefficientRidge:
        """Fit the quadratic ridge with penalty 1 and an unpenalised intercept."""
        keep = np.ptp(inputs, axis=0) != 0
        active = inputs[:, keep]
        centre = active.mean(axis=0)
        scale = active.std(axis=0)
        all_terms = _quadratic_terms((active - centre) / scale)
        term_mask = np.ptp(all_terms, axis=0) != 0
        terms = all_terms[:, term_mask]
        terms_mean = terms.mean(axis=0)
        targets_mean = targets.mean(axis=0)
        centred = terms - terms_mean
        weights = np.linalg.solve(
            centred.T @ centred + np.eye(centred.shape[1]),
            centred.T @ (targets - targets_mean),
        )
        return cls(
            keep,
            centre,
            scale,
            term_mask,
            targets_mean - terms_mean @ weights,
            weights,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        z = (inputs[..., self.input_mask] - self.centre) / self.scale
        terms = _quadratic_terms(np.atleast_2d(z))[:, self.term_mask]
        predictions = terms @ self.weights + self.intercept
        return predictions[0] if inputs.ndim == 1 else predictions


@dataclass(frozen=True)
class OriginalBaselines:
    regional_means: np.ndarray  # Off-wall, wall in dimensional field units.
    coefficients: dict[str, CoefficientRidge]

    def predict_fields(self, surf: np.ndarray) -> np.ndarray:
        return self.regional_means[surf.astype(np.intp)]

    def predict_coefficients(self, case_id: str) -> np.ndarray:
        family, inputs = naca_design(case_id)
        return self.coefficients[family].predict(inputs)

    def to_dict(self) -> dict:
        return {
            "regional_means": self.regional_means.tolist(),
            "coefficients": {
                family: {
                    field: value.tolist() for field, value in asdict(model).items()
                }
                for family, model in self.coefficients.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> OriginalBaselines:
        models = {}
        for family, record in data["coefficients"].items():
            arrays = {key: np.asarray(value) for key, value in record.items()}
            arrays["input_mask"] = arrays["input_mask"].astype(bool)
            arrays["term_mask"] = arrays["term_mask"].astype(bool)
            arrays["weights"] = arrays["weights"].reshape(-1, len(arrays["intercept"]))
            models[family] = CoefficientRidge(**arrays)
        return cls(np.asarray(data["regional_means"]), models)


def fit_staged_baselines(train_stage: Path, pin: dict, out: Path) -> None:
    rows, stage_digest = read_complete_partition(train_stage, "train", pin)
    if out.exists():
        raise FileExistsError(out)
    sums = np.zeros((2, len(TARGET_NAMES)), dtype=np.float64)
    designs: dict[str, list[np.ndarray]] = {"naca4": [], "naca5": []}
    coefficients: dict[str, list[np.ndarray]] = {"naca4": [], "naca5": []}
    for row in rows:
        case_id = row["id"]
        family, design = naca_design(case_id)
        inlet, angle = map(float, case_id.split("_")[2:4])
        mesh = load_original_mesh(*verified_mesh_paths(train_stage, row))
        sim = OriginalForceSimulation(mesh, inlet, np.deg2rad(angle))
        target = force_coefficients(sim, reference=True)[:, 0]
        if not np.isfinite(target).all():
            raise ValueError(f"non-finite reference coefficients for {case_id}")
        features = sim.features
        sums[0] += features.y[~features.surf].mean(axis=0)
        sums[1] += features.y[features.surf].mean(axis=0)
        designs[family].append(design)
        coefficients[family].append(target)

    baseline = OriginalBaselines(
        sums / len(rows),
        {
            family: CoefficientRidge.fit(np.stack(designs[family]), np.stack(values))
            for family, values in coefficients.items()
        },
    )
    with out.open("x") as stream:
        json.dump(
            baseline.to_dict()
            | {
                "schema": 1,
                "field_names": TARGET_NAMES,
                "coefficient_names": ("CL", "CD"),
                "fit_stage_sha256": stage_digest,
                "fit_cases": [row["id"] for row in rows],
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_stage", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit_staged_baselines(args.train_stage, json.loads(PIN.read_text()), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
