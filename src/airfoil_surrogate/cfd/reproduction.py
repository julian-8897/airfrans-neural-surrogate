"""Select the original NACA reproduction cases and qualify completed CFD outputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from airfoil_surrogate.cfd.reference_mesh import (
    approved_reference_mesh,
    verify_reference_mesh,
)
from airfoil_surrogate.data.original import OriginalMesh, load_original_mesh
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import (
    case_order_sha256,
    read_complete_partition,
    verified_mesh_paths,
)
from airfoil_surrogate.eval.forces import OriginalForceSimulation, force_coefficients
from airfoil_surrogate.eval.original_baselines import naca_design

GENERATOR_REVISION = "2dbf00c5ecbb8d6cef720db1e72c1fe40307eeab"
OPERATING_TARGETS = ((2.25e6, -4, "naca4"), (4e6, 5, "naca5"), (5.75e6, 12, "naca4"))
CP_STATIONS = np.linspace(0.01, 0.99, 99)
WINDOW = 500


def select_reproduction_cases(train_ids: list[str]) -> list[dict]:
    candidates = [(case_id, *naca_design(case_id)) for case_id in train_ids]
    selected = []
    for reynolds, angle, family in OPERATING_TARGETS:
        case_id, _, design = min(
            (entry for entry in candidates if entry[1] == family),
            key=lambda entry: (
                ((entry[2][-2] - reynolds) / 1e6) ** 2
                + ((entry[2][-1] - angle) / 5) ** 2,
                entry[0],
            ),
        )
        speed = float(case_id.split("_")[2])
        aoa_deg = float(design[-1])
        selected.append(
            {
                "id": case_id,
                "family": family,
                "naca_parameters": design[:-2].tolist(),
                "inlet_velocity": speed,
                "reynolds": float(design[-2]),
                "aoa_deg": aoa_deg,
                "iteration_limit": 40_000 if abs(aoa_deg) > 10 else 20_000,
            }
        )
        candidates = [entry for entry in candidates if entry[0] != case_id]
    return selected


def _make_plan(splits: Path, pin: dict) -> dict:
    raw = splits.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin["split_sha256"]:
        raise ValueError("original AirfRANS split checksum mismatch")
    train = json.loads(raw)["full_train"][: pin["partitions"]["train"]["size"]]
    if case_order_sha256(train) != pin["partitions"]["train"]["order_sha256"]:
        raise ValueError("original AirfRANS training order mismatch")
    return {
        "schema": 1,
        "generator_revision": GENERATOR_REVISION,
        "official_split_sha256": digest,
        "cases": select_reproduction_cases(train),
    }


def read_convergence(coefficients: str, solver_log: str, limit: int) -> dict:
    """Check the final two contiguous 500-iteration force windows."""
    times = re.findall(r"^\s*Time\s*=\s*([0-9.eE+-]+)\s*$", solver_log, flags=re.M)
    if (
        not times
        or "FOAM FATAL" in solver_log
        or not re.search(r"^\s*End\s*$", solver_log, flags=re.M)
    ):
        raise ValueError("simpleFoam did not record a completed solve")
    final_time = float(times[-1])
    header = [
        line[1:].split() for line in coefficients.splitlines() if line.startswith("#")
    ]
    fields = next((columns for columns in header if columns[:1] == ["Time"]), [])
    if "Cl" not in fields or "Cd" not in fields:
        raise ValueError("force monitor must contain Time, Cl and Cd columns")
    data = np.loadtxt(
        io.StringIO(coefficients),
        usecols=(0, fields.index("Cl"), fields.index("Cd")),
        ndmin=2,
    )
    tail = data[-2 * WINDOW :]
    if (
        not 2 * WINDOW <= final_time <= limit
        or len(tail) != 2 * WINDOW
        or not np.isfinite(tail).all()
        or not np.array_equal(
            tail[:, 0], np.arange(final_time - 2 * WINDOW + 1, final_time + 1)
        )
    ):
        raise ValueError(
            "force monitor does not cover the final 1,000 consecutive iterations"
        )
    previous = tail[:WINDOW, 1:].mean(axis=0)
    final = tail[WINDOW:, 1:].mean(axis=0)
    delta = np.abs(final - previous)
    tolerance = np.array([0.002, 0.00005])
    return {
        "passed": bool(np.all(delta <= tolerance)),
        "window_length": WINDOW,
        "last_iteration": int(final_time),
        "previous_mean": previous.tolist(),
        "last_mean": final.tolist(),
        "absolute_difference": delta.tolist(),
        "tolerance": tolerance.tolist(),
    }


def mesh_check_passed(log: str) -> bool:
    return bool(re.search(r"^\s*Mesh OK\.\s*$", log, re.M)) and not bool(
        re.search(r"Failed \d+ mesh checks|FOAM FATAL", log)
    )


def _wall_paths(x: np.ndarray, lines: np.ndarray) -> list[np.ndarray]:
    if len(lines) % 3 or np.any(lines[::3] != 2):
        raise ValueError("surface pressure comparison requires two-point wall edges")
    neighbours: list[list[int]] = [[] for _ in x]
    for a, b in lines.reshape(-1, 3)[:, 1:]:
        neighbours[a].append(b)
        neighbours[b].append(a)
    if any(len(points) != 2 or points[0] == points[1] for points in neighbours):
        raise ValueError("surface pressure comparison requires a closed wall")
    leading, trailing = int(x.argmin()), int(x.argmax())
    paths = []
    for first in neighbours[leading]:
        branch = [leading]
        node = first
        while node != trailing:
            previous = branch[-1]
            branch.append(node)
            node = next(
                adjacent for adjacent in neighbours[node] if adjacent != previous
            )
            if node == leading:
                raise ValueError(
                    "leading and trailing edges lie on different wall loops"
                )
        paths.append(np.asarray([*branch, trailing]))
    if len(paths[0]) + len(paths[1]) != len(x) + 2:
        raise ValueError("surface pressure comparison requires one wall loop")
    return paths


def pressure_profiles(
    mesh: OriginalMesh, pressure: np.ndarray, inlet: float
) -> np.ndarray:
    """Interpolate upper and lower Cp at identical stations in the unit chord."""
    x, y = np.asarray(mesh.aerofoil.points[:, :2], dtype=np.float64).T
    profiles = []
    for branch in _wall_paths(x, mesh.aerofoil.lines):
        chord = x[branch]
        relevant = (chord[1:] >= CP_STATIONS[0]) & (chord[:-1] <= CP_STATIONS[-1])
        if (
            chord[0] > CP_STATIONS[0]
            or chord[-1] < CP_STATIONS[-1]
            or np.any(np.diff(chord)[relevant] <= 0)
        ):
            raise ValueError("wall branch does not span the Cp stations monotonically")
        # Keep the interval bracketing the stations. Nose-only folds cannot
        # influence the interpolant over the comparison region.
        edges = np.flatnonzero(relevant)
        branch = branch[edges[0] : edges[-1] + 2]
        height = np.interp(CP_STATIONS, x[branch], y[branch])
        cp = np.interp(CP_STATIONS, x[branch], pressure[branch]) / (0.5 * inlet**2)
        profiles.append((height, cp))
    upper, lower = sorted(profiles, key=lambda item: item[0].mean(), reverse=True)
    if np.any(upper[0] <= lower[0]):
        raise ValueError("upper and lower wall branches cross at a Cp station")
    return np.stack((upper[1], lower[1]))


def compare_reproduction(
    case: dict,
    original: OriginalMesh,
    reproduced: OriginalMesh,
    checkmesh_log: str,
    solver_log: str,
    coefficients: str,
    *,
    reference_mesh: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    inlet, angle = case["inlet_velocity"], np.deg2rad(case["aoa_deg"])
    native = [
        OriginalForceSimulation(mesh, inlet, angle) for mesh in (original, reproduced)
    ]
    original_force, reproduced_force = [
        force_coefficients(sim, reference=True) for sim in native
    ]
    original_cp, reproduced_cp = [
        pressure_profiles(mesh, sim.features.y[mesh.wall_nodes, 2], inlet)
        for mesh, sim in zip((original, reproduced), native, strict=True)
    ]
    if not all(
        np.isfinite(values).all()
        for values in (original_force, reproduced_force, original_cp, reproduced_cp)
    ):
        raise ValueError("non-finite force or surface pressure in reproduction")
    strict_mesh_ok = mesh_check_passed(checkmesh_log)
    mesh_ok = strict_mesh_ok or reference_mesh is not None
    delta = np.abs(reproduced_force[:, 0] - original_force[:, 0])
    tolerance = np.maximum([0.02, 0.0005], np.abs(original_force[:, 0]) * [0.03, 0.05])
    cp_rmse = np.sqrt(np.mean((reproduced_cp - original_cp) ** 2, axis=1))
    convergence = read_convergence(coefficients, solver_log, case["iteration_limit"])
    return (
        {
            "passed": bool(
                mesh_ok
                and convergence["passed"]
                and np.all(delta <= tolerance)
                and np.all(cp_rmse <= 0.05)
            ),
            "case": case["id"],
            "coefficient_names": ["CL", "CD"],
            "contribution_names": ["total", "pressure", "viscous"],
            "mesh_ok": mesh_ok,
            "strict_mesh_ok": strict_mesh_ok,
            "original_coefficients": original_force.tolist(),
            "reproduced_coefficients": reproduced_force.tolist(),
            "coefficient_absolute_difference": delta.tolist(),
            "coefficient_tolerance": tolerance.tolist(),
            "cp_sides": ["upper", "lower"],
            "cp_rmse": cp_rmse.tolist(),
            "cp_rmse_tolerance": 0.05,
            "convergence": convergence,
        }
        | ({"reference_mesh_exception": reference_mesh} if reference_mesh else {}),
        original_cp,
        reproduced_cp,
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _overlay(path: Path, original: np.ndarray, reproduced: np.ndarray) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 3.5), layout="constrained")
    for i, (axis, side) in enumerate(
        zip(figure.subplots(1, 2), ("Upper", "Lower"), strict=True)
    ):
        axis.plot(CP_STATIONS, original[i], label="AirfRANS")
        axis.plot(CP_STATIONS, reproduced[i], linestyle="--", label="Reproduction")
        axis.set(xlabel="x/c", ylabel="Cp", title=side)
        axis.invert_yaxis()
        axis.legend()
    figure.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("official_splits", type=Path)
    plan.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("plan", type=Path)
    verify.add_argument("train_stage", type=Path)
    verify.add_argument("case_id")
    for argument in (
        "internal",
        "aerofoil",
        "checkmesh_log",
        "solver_log",
        "force_history",
    ):
        verify.add_argument(argument, type=Path)
    verify.add_argument("--out", type=Path, required=True)
    verify.add_argument("--reference-mesh-exception", action="store_true")
    args = parser.parse_args()
    pin = json.loads(PIN.read_text())
    if args.command == "plan":
        _write_json(args.out, _make_plan(args.official_splits, pin))
        return

    data = args.plan.read_bytes()
    rows, train_digest = read_complete_partition(args.train_stage, "train", pin)
    expected = {
        "schema": 1,
        "generator_revision": GENERATOR_REVISION,
        "official_split_sha256": pin["split_sha256"],
        "cases": select_reproduction_cases([row["id"] for row in rows]),
    }
    if json.loads(data) != expected:
        raise ValueError("unrecognised reproduction plan")
    case = {row["id"]: row for row in expected["cases"]}[args.case_id]
    source = {row["id"]: row for row in rows}[args.case_id]
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "plan_sha256": hashlib.sha256(data).hexdigest(),
        "train_stage_sha256": train_digest,
        "original_mesh": source["files"],
    }
    try:
        original = load_original_mesh(*verified_mesh_paths(args.train_stage, source))
        reproduced = load_original_mesh(args.internal, args.aerofoil)
        logs = {}
        for field in ("checkmesh_log", "solver_log", "force_history"):
            raw = getattr(args, field).read_bytes()
            report[f"{field}_sha256"] = hashlib.sha256(raw).hexdigest()
            logs[field] = raw.decode()
        for field in ("internal", "aerofoil"):
            with getattr(args, field).open("rb") as stream:
                report[f"{field}_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        reference_mesh = None
        if args.reference_mesh_exception:
            reference_mesh = approved_reference_mesh(case["id"], report["plan_sha256"])
            verify_reference_mesh(
                logs["checkmesh_log"],
                args.checkmesh_log.parent / "constant/polyMesh",
                reference_mesh,
            )
        result, original_cp, reproduced_cp = compare_reproduction(
            case,
            original,
            reproduced,
            logs["checkmesh_log"],
            logs["solver_log"],
            logs["force_history"],
            reference_mesh=reference_mesh,
        )
    except (ValueError, OSError) as error:
        _write_json(
            args.out / "reproduction.json",
            report | {"passed": False, "case": case["id"], "error": str(error)},
        )
        raise
    np.savetxt(
        args.out / "surface-cp.csv",
        np.column_stack((CP_STATIONS, original_cp.T, reproduced_cp.T)),
        delimiter=",",
        header="x_over_c,original_upper,original_lower,reproduced_upper,reproduced_lower",
        comments="",
    )
    _overlay(args.out / "surface-cp.png", original_cp, reproduced_cp)
    _write_json(args.out / "reproduction.json", report | result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
