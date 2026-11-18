"""Evaluate a frozen Transolver and its baselines on the original full-task test set."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from airfoil_surrogate.data.features import TARGET_NAMES, NodeFeatures, Normaliser
from airfoil_surrogate.data.original import load_original_mesh
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import (
    read_complete_partition,
    verified_mesh_paths,
)
from airfoil_surrogate.eval.forces import (
    OriginalForceSimulation,
    coefficient_relative_error,
    coefficient_spearman,
    force_coefficients,
)
from airfoil_surrogate.eval.metrics import field_errors, json_values, summarise
from airfoil_surrogate.eval.original_baselines import OriginalBaselines
from airfoil_surrogate.models.transolver import Transolver, TransolverConfig
from airfoil_surrogate.train import git_state, pick_device

BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 10_000


def _frozen_inputs(run: Path, baselines: Path, frozen: dict) -> dict[str, bytes]:
    paths = {
        name: run / name
        for name in (
            "run.json",
            "selection.json",
            "normaliser.json",
            "metrics.jsonl",
            "model.pt",
        )
    }
    paths["baselines"] = baselines
    if frozen["schema"] != 1 or paths.keys() != frozen["sha256"].keys():
        raise ValueError("invalid evaluation freeze manifest")
    inputs = {}
    for name, path in paths.items():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != frozen["sha256"][name]:
            raise ValueError(f"frozen evaluation input changed: {name}")
        inputs[name] = data
    return inputs


@torch.inference_mode()
def predict_native_fields(
    model: Transolver, features: NodeFeatures, norm: Normaliser, device: torch.device
) -> np.ndarray:
    """Return dimensional predictions from one forward on the complete mesh."""
    x = torch.as_tensor(norm.inputs(features.x), dtype=torch.float32, device=device)
    pos = torch.as_tensor(features.pos, dtype=torch.float32, device=device)
    result = model(x[None], pos[None])[0].cpu().numpy().astype(np.float64)
    fields = norm.inverse_targets(result)
    if fields.shape != features.y.shape or not np.isfinite(fields).all():
        raise ValueError("invalid native-mesh predictions")
    return fields


def _coefficient_errors(prediction: np.ndarray, reference: np.ndarray) -> dict:
    return {
        "coefficient_absolute_error": np.abs(prediction - reference),
        "coefficient_relative_error": coefficient_relative_error(prediction, reference),
    }


def score_native_fields(
    sim: OriginalForceSimulation, fields: np.ndarray, norm: Normaliser
) -> tuple[dict, np.ndarray]:
    """Score raw fields and integrate coefficients with and without wall no-slip."""
    features = sim.features
    metrics = field_errors(norm.targets(fields), features.y, features.surf, norm)
    reference = force_coefficients(sim, reference=True)
    reference_shear = sim.wallshearstress(over_airfoil=True, reference=True)
    sim.set_fields(fields)
    raw = force_coefficients(sim)
    corrected = fields.copy()
    corrected[features.surf, :2] = 0.0
    sim.set_fields(corrected)
    predicted = force_coefficients(sim)
    shear = sim.wallshearstress(over_airfoil=True)
    if not all(
        np.isfinite(value).all()
        for value in (reference, raw, predicted, shear, reference_shear)
    ):
        raise ValueError("non-finite native-mesh force integration")
    return {
        **metrics,
        **_coefficient_errors(predicted[:, 0], reference[:, 0]),
        "coefficients": predicted,
        "raw_coefficients": raw,
        "wall_pressure_rmse": np.sqrt(
            ((fields[features.surf, 2] - features.y[features.surf, 2]) ** 2).mean()
        ),
        "wall_shear_rmse": np.sqrt(((shear - reference_shear) ** 2).mean(axis=0)),
    }, reference


def paired_mean_differences(delta: np.ndarray, resamples: np.ndarray) -> list[dict]:
    """Calculate percentile intervals on paired case differences for each channel."""
    result = []
    for values in delta.T:
        finite = np.isfinite(values)
        picked = values[resamples]
        defined = np.isfinite(picked)
        counts = defined.sum(axis=1)
        totals = np.where(defined, picked, 0.0).sum(axis=1)
        draws = totals[counts > 0] / counts[counts > 0]
        interval = np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) else None
        result.append(
            {
                "mean": float(values[finite].mean()) if finite.any() else None,
                "interval95": interval,
                "n_cases": int(finite.sum()),
                "n_draws": len(draws),
                "supports_improvement": bool(interval is not None and interval[1] < 0),
            }
        )
    return result


def summarise_official(rows: list[dict]) -> dict:
    common = (
        "volume_mse",
        "surface_mse",
        "volume_relative_l2",
        "surface_relative_l2",
        "coefficient_absolute_error",
        "coefficient_relative_error",
    )

    def column(model: str, name: str) -> np.ndarray:
        return np.asarray([row[model][name] for row in rows], dtype=np.float64)

    reference = np.asarray([row["reference_coefficients"] for row in rows])[:, :, 0]
    resamples = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        len(rows), size=(BOOTSTRAP_DRAWS, len(rows))
    )
    summary = {
        "case_count": len(rows),
        "fields": TARGET_NAMES,
        "coefficients": ("CL", "CD"),
        "force_components": ("total", "pressure", "viscous"),
        "metrics": {
            model: {name: summarise(column(model, name)) for name in common}
            for model in ("transolver", "baseline")
        },
        "spearman": {
            "transolver": coefficient_spearman(
                column("transolver", "coefficients")[:, :, 0], reference
            ),
            "transolver_raw": coefficient_spearman(
                column("transolver", "raw_coefficients")[:, :, 0], reference
            ),
            "baseline": coefficient_spearman(
                column("baseline", "coefficients"), reference
            ),
        },
        "paired_transolver_minus_baseline": {
            name: paired_mean_differences(
                column("transolver", name) - column("baseline", name), resamples
            )
            for name in common
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    summary["metrics"]["transolver"]["wall_pressure_rmse"] = summarise(
        column("transolver", "wall_pressure_rmse")[:, None]
    )
    summary["metrics"]["transolver"]["wall_shear_rmse"] = summarise(
        column("transolver", "wall_shear_rmse")
    )
    return summary


def rebuild_summary(evaluation: Path, out: Path) -> None:
    expected = json.loads((evaluation / "evaluation.json").read_text())["case_ids"]
    rows = [
        json.loads(line)
        for line in (evaluation / "cases.jsonl").read_text().splitlines()
    ]
    if not expected or [row["case"] for row in rows] != expected:
        raise ValueError("incomplete or out-of-order held-out results")
    with out.open("x") as stream:
        json.dump(summarise_official(rows), stream, indent=2, allow_nan=False)
        stream.write("\n")


def evaluate_official(
    run: Path,
    baselines: Path,
    freeze_path: Path,
    test_stage: Path,
    pin: dict,
    out: Path,
    device: torch.device,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    frozen_bytes = freeze_path.read_bytes()
    frozen_digest = hashlib.sha256(frozen_bytes).hexdigest()
    frozen = json.loads(frozen_bytes)
    inputs = _frozen_inputs(run, baselines, frozen)
    norm = Normaliser.from_dict(json.loads(inputs["normaliser.json"]))
    baseline = OriginalBaselines.from_dict(json.loads(inputs["baselines"]))
    checkpoint = torch.load(
        io.BytesIO(inputs["model.pt"]), map_location="cpu", weights_only=True
    )
    model = Transolver(TransolverConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    rows, test_digest = read_complete_partition(
        test_stage, "test", pin, frozen_sha256=frozen_digest
    )

    out.mkdir(parents=True, exist_ok=False)
    predictions = out / "predictions"
    predictions.mkdir()
    (out / "evaluation.json").write_text(
        json.dumps(
            {
                "split": "test",
                "case_ids": [row["id"] for row in rows],
                "test_stage_sha256": test_digest,
                "frozen_sha256": frozen_digest,
                "selected_checkpoint": frozen["sha256"]["model.pt"],
                "git": git_state(),
                "versions": {
                    name: version(name)
                    for name in ("torch", "airfrans", "vtk", "pyvista")
                },
                "device": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device),
            },
            indent=2,
        )
        + "\n"
    )
    with (out / "cases.jsonl").open("x") as log:
        for case in rows:
            case_id = case["id"]
            inlet, angle = map(float, case_id.split("_")[2:4])
            mesh = load_original_mesh(*verified_mesh_paths(test_stage, case))
            sim = OriginalForceSimulation(mesh, inlet, np.deg2rad(angle))
            features = sim.features
            fields = predict_native_fields(model, features, norm, device)
            model_metrics, reference = score_native_fields(sim, fields, norm)
            baseline_fields = baseline.predict_fields(features.surf)
            baseline_coeffs = baseline.predict_coefficients(case_id)
            baseline_metrics = {
                **field_errors(
                    norm.targets(baseline_fields), features.y, features.surf, norm
                ),
                **_coefficient_errors(baseline_coeffs, reference[:, 0]),
                "coefficients": baseline_coeffs,
            }
            path = predictions / f"{case_id}.npz"
            with path.open("xb") as stream:
                np.savez(stream, fields=fields)
            with path.open("rb") as stream:
                prediction_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            row = {
                "case": case_id,
                "mesh_files": case["files"],
                "prediction_file": str(path.relative_to(out)),
                "prediction_sha256": prediction_sha256,
                "nodes": len(fields),
                "reference_coefficients": reference.tolist(),
                "transolver": {
                    name: json_values(value) for name, value in model_metrics.items()
                },
                "baseline": {
                    name: json_values(value) for name, value in baseline_metrics.items()
                },
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    rebuild_summary(out, out / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("training_run", type=Path)
    evaluate.add_argument("baselines", type=Path)
    evaluate.add_argument("frozen", type=Path)
    evaluate.add_argument("test_stage", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    summarise_saved = commands.add_parser("summarise")
    summarise_saved.add_argument("evaluation", type=Path)
    summarise_saved.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarise":
        rebuild_summary(args.evaluation, args.out)
    else:
        torch.set_float32_matmul_precision("highest")
        evaluate_official(
            args.training_run,
            args.baselines,
            args.frozen,
            args.test_stage,
            json.loads(PIN.read_text()),
            args.out,
            pick_device(args.device),
        )
    print(args.out)


if __name__ == "__main__":
    main()
