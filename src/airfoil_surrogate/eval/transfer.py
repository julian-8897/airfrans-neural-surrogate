"""Evaluate the frozen Transolver on the matched, CFD-qualified cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from airfoil_surrogate.cfd.campaign import QUOTAS
from airfoil_surrogate.data.original import load_original_mesh
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.eval.forces import AIRFRANS_VISCOSITY, OriginalForceSimulation
from airfoil_surrogate.eval.metrics import json_values
from airfoil_surrogate.eval.original import (
    frozen_inputs,
    load_frozen_model,
    predict_native_fields,
    score_native_fields,
)
from airfoil_surrogate.eval.transfer_metrics import summarise_transfer, weighted_mse
from airfoil_surrogate.train import git_state, pick_device


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _linked_json(record: dict) -> dict:
    data = Path(record["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"frozen CFD evidence changed: {record['path']}")
    return json.loads(data)


def frozen_cases(cohorts: dict, pin: dict) -> list[dict]:
    if (
        cohorts.get("schema") != 1
        or cohorts.get("budget", {}).get("within_cap") is not True
    ):
        raise ValueError("match evaluation requires a budget-checked CFD cohort freeze")
    cases = cohorts["cases"]
    ids = [case["id"] for case in cases]
    quotas = Counter((case["group"], case.get("family")) for case in cases)
    if (
        quotas != QUOTAS
        or len(set(ids)) != len(cases)
        or any(
            not case or case in {".", ".."} or Path(case).name != case for case in ids
        )
    ):
        raise ValueError("invalid frozen cohort membership")
    operating_points = {
        (
            case["inlet_velocity"],
            case["reynolds"],
            case["aoa_deg"],
            case["iteration_limit"],
        )
        for case in cases
    }
    if operating_points != {(4e6 * AIRFRANS_VISCOSITY, 4e6, 4.0, 20_000)}:
        raise ValueError("frozen cohorts do not share the specified operating point")
    gate = _linked_json(cohorts["qualification"])
    candidate = _linked_json(cohorts["candidate_manifest"])
    if (
        gate["passed"] is not True
        or gate["official_split_sha256"] != pin["split_sha256"]
        or candidate["sources"]["airfrans_split_sha256"] != pin["split_sha256"]
    ):
        raise ValueError(
            "frozen CFD cohorts require qualification against the pinned split"
        )
    return cases


def _verified_mesh_paths(case: dict) -> tuple[Path, Path]:
    paths = []
    for name in ("internal.vtu", "aerofoil.vtp"):
        record = case["native"][name]
        path = Path(record["path"])
        if path.stat().st_size != record["size"] or _sha(path) != record["sha256"]:
            raise ValueError(f"frozen native mesh changed: {case['id']} {name}")
        paths.append(path)
    return paths[0], paths[1]


def rebuild_summary(evaluation: Path, out: Path) -> None:
    metadata = json.loads((evaluation / "evaluation.json").read_text())
    rows = [
        json.loads(line)
        for line in (evaluation / "cases.jsonl").read_text().splitlines()
    ]
    if (
        metadata["split"] != "matched_transfer"
        or [[row["case"], row["group"]] for row in rows] != metadata["cases"]
        or Counter(row["group"] for row in rows) != {"uiuc": 30, "naca": 30}
    ):
        raise ValueError("incomplete or out-of-order matched-cohort results")
    summary, rankings = summarise_transfer(rows)
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "summary.json", summary)
    _write(out / "rankings.json", rankings)


def evaluate_transfer(
    run: Path,
    baselines: Path,
    freeze_path: Path,
    cohort_path: Path,
    pin: dict,
    out: Path,
    device: torch.device,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    frozen_bytes = freeze_path.read_bytes()
    cohort_bytes = cohort_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    cases = frozen_cases(json.loads(cohort_bytes), pin)
    meshes = [_verified_mesh_paths(case) for case in cases]
    model, norm = load_frozen_model(frozen_inputs(run, baselines, frozen), device)

    out.mkdir(parents=True, exist_ok=False)
    predictions = out / "predictions"
    predictions.mkdir()
    _write(
        out / "evaluation.json",
        {
            "split": "matched_transfer",
            "cases": [[case["id"], case["group"]] for case in cases],
            "operating_point": {
                key: cases[0][key] for key in ("inlet_velocity", "aoa_deg", "reynolds")
            },
            "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
            "frozen_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
            "selected_checkpoint": frozen["sha256"]["model.pt"],
            "git": git_state(),
            "versions": {
                name: version(name)
                for name in ("torch", "airfrans", "numpy", "scipy", "vtk", "pyvista")
            },
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
        },
    )
    with (out / "cases.jsonl").open("x") as log:
        for case, paths in zip(cases, meshes, strict=True):
            mesh = load_original_mesh(*paths)
            sim = OriginalForceSimulation(
                mesh, case["inlet_velocity"], np.deg2rad(case["aoa_deg"])
            )
            fields = predict_native_fields(model, sim.features, norm, device)
            scores, reference = score_native_fields(sim, fields, norm)
            scores.update(
                weighted_mse(mesh, norm.targets(fields) - norm.targets(sim.features.y))
            )
            path = predictions / f"{case['id']}.npz"
            with path.open("xb") as prediction:
                np.savez(prediction, fields=fields)
            row = {
                "case": case["id"],
                "group": case["group"],
                "family": case.get("family"),
                "cfd_run_id": case["run_id"],
                "cfd_job_id": case["job_id"],
                "mesh_files": case["native"],
                "prediction": {
                    "path": path.relative_to(out).as_posix(),
                    "sha256": _sha(path),
                },
                "reference_coefficients": json_values(reference),
                "transolver": {
                    name: json_values(value) for name, value in scores.items()
                },
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    rebuild_summary(out, out / "report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("training_run", type=Path)
    evaluate.add_argument("baselines", type=Path)
    evaluate.add_argument("frozen", type=Path)
    evaluate.add_argument("cohorts", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    summarise = commands.add_parser("summarise")
    summarise.add_argument("evaluation", type=Path)
    summarise.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        evaluate_transfer(
            args.training_run,
            args.baselines,
            args.frozen,
            args.cohorts,
            json.loads(PIN.read_text()),
            args.out,
            pick_device(args.device),
        )
    else:
        rebuild_summary(args.evaluation, args.out)


if __name__ == "__main__":
    main()
