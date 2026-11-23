"""Run one approved OpenFOAM case using the pinned AirfRANS case setup."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from contextlib import chdir
from pathlib import Path

import numpy as np

from airfoil_surrogate.cfd.accounting import CORES
from airfoil_surrogate.cfd.campaign import check_qualification, verify_case_outputs
from airfoil_surrogate.cfd.convert import convert_files
from airfoil_surrogate.cfd.reference_mesh import (
    approved_reference_mesh,
    verify_reference_mesh,
)
from airfoil_surrogate.cfd.reproduction import (
    GENERATOR_REVISION,
    mesh_check_passed,
    read_convergence,
)

TEMPLATE = Path("Simulations/airFoil2DInit")
GENERATOR_FILES = (
    "simulation_generator.py",
    "naca_generator.py",
    "params.yaml",
    str(TEMPLATE),
)
MONITOR_PATH = Path("postProcessing/forceCoeffs1/0/coefficient.dat")


def _generator_archive(repo: Path, out: Path) -> str:
    """Materialise the pinned Git objects, independent of the checkout worktree."""
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            GENERATOR_REVISION,
            *GENERATOR_FILES,
        ]
    )
    out.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(out, filter="data")
    return hashlib.sha256(archive).hexdigest()


def per_iteration_monitor(control: str) -> str:
    """Write the existing force monitor each iteration without writing fields."""
    pattern = re.compile(
        r"(^[ \t]*forceCoeffs1[ \t]*\n[ \t]*\{\n)(.*?)(^[ \t]*\}[ \t]*$)",
        re.M | re.S,
    )
    matches = list(pattern.finditer(control))
    if len(matches) != 1:
        raise ValueError(
            "expected one forceCoeffs1 object in the generated controlDict"
        )
    match = matches[0]
    body = match[2]
    for setting, value in (("writeInterval", "1"), ("writeFields", "no")):
        body, count = re.subn(
            rf"(^[ \t]*{setting}[ \t]+)[^;\n]+;", rf"\g<1>{value};", body, flags=re.M
        )
        if count != 1:
            raise ValueError(f"expected one {setting} in forceCoeffs1")
    return control[: match.start(2)] + body + control[match.end(2) :]


def _external_geometry(case: dict, plan_dir: Path) -> np.ndarray:
    raw = (plan_dir / case["geometry_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["geometry_sha256"]:
        raise ValueError("case geometry checksum mismatch")
    return np.loadtxt(io.BytesIO(raw), delimiter=",", ndmin=2)


def _setup_case(source: Path, case: dict, plan_dir: Path, target: Path) -> None:
    """Use the pinned setup functions with either source of wall coordinates."""
    if ("naca_parameters" in case) == ("geometry_file" in case):
        raise ValueError("provide NACA parameters or a pinned geometry file")
    inlet, aoa = case["inlet_velocity"], case["aoa_deg"]
    limit = case["iteration_limit"]
    if (
        not np.isfinite([inlet, aoa]).all()
        or inlet <= 0
        or limit != (40_000 if abs(aoa) > 10 else 20_000)
    ):
        raise ValueError("invalid inlet conditions or iteration limit")

    # This entry point runs one case per process. The upstream module reads
    # params.yaml on import and imports its sibling naca_generator by name.
    sys.path.insert(0, str(source))
    try:
        with chdir(source):
            generator = importlib.import_module("simulation_generator")
            if Path(generator.__file__).resolve() != source / "simulation_generator.py":
                raise RuntimeError("unexpected simulation_generator module location")
            params = generator.params
            if "naca_parameters" in case:
                geometry = generator.naca_generator(
                    case["naca_parameters"],
                    nb_samples=2000,
                    scale=1,
                    origin=(0, 0),
                    cosine_spacing=True,
                    verbose=False,
                    CTE=True,
                )
            else:
                geometry = _external_geometry(case, plan_dir)
            if (
                geometry.ndim != 2
                or geometry.shape[1] != 2
                or len(geometry) < 5
                or not np.isfinite(geometry).all()
                or not np.array_equal(geometry[[0, -1]], [[1, 0], [1, 0]])
            ):
                raise ValueError(
                    "expected a finite contour with a closed unit-chord trailing edge"
                )

            shutil.copytree(source / TEMPLATE, target)
            path = str(target) + os.sep
            grading = generator.dict_grading(
                *(
                    params[key]
                    for key in ("L", "y_h", "y_hd", "x_h", "y_exp", "x_exp", "x_expd")
                ),
                aoa,
                geometry,
            )
            generator.blockMeshDict_generator(path, geometry, grading)
            generator.system_generator(
                path, inlet, aoa, CORES, n_iter=limit, compressible=False
            )
            generator.init_generator(
                path, inlet, aoa, 2 * params["L"], "SST", params["y_h"], False
            )
            np.savetxt(target / "geometry.csv", geometry, delimiter=",")
    finally:
        sys.path.pop(0)

    control = target / "system/controlDict"
    control.write_text(per_iteration_monitor(control.read_text()))


def _command(case: Path, log: str, args: list[str]) -> None:
    with (case / log).open("x") as stream:
        subprocess.run(
            args, cwd=case, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def solve_case(
    case_dir: Path, iteration_limit: int, *, reference_mesh: dict | None = None
) -> tuple[Path, Path, dict]:
    _command(case_dir, "log.blockMesh", ["blockMesh"])
    _command(case_dir, "log.checkMesh", ["checkMesh"])
    mesh_log = (case_dir / "log.checkMesh").read_text()
    if reference_mesh is not None:
        verify_reference_mesh(mesh_log, case_dir / "constant/polyMesh", reference_mesh)
    elif not mesh_check_passed(mesh_log):
        raise ValueError("checkMesh failed")
    _command(case_dir, "log.decomposePar", ["decomposePar"])
    _command(
        case_dir,
        "log.simpleFoam",
        ["mpirun", "-np", str(CORES), "simpleFoam", "-parallel"],
    )
    convergence = read_convergence(
        (case_dir / MONITOR_PATH).read_text(),
        (case_dir / "log.simpleFoam").read_text(),
        iteration_limit,
    )
    if not convergence["passed"]:
        raise ValueError(f"force monitor failed convergence: {convergence}")
    _command(case_dir, "log.reconstructPar", ["reconstructPar", "-latestTime"])
    _command(
        case_dir,
        "log.foamToVTK",
        ["foamToVTK", "-noZero", "-latestTime", "-fields", "(p U nut wallShearStress)"],
    )
    volumes = list((case_dir / "VTK").rglob("internal.vtu"))
    if len(volumes) != 1:
        raise ValueError("expected one volume export at the final solver time")
    wall = volumes[0].parent / "boundary/aerofoil.vtp"
    return volumes[0], wall, convergence


def _write_json(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retain_completed_case(
    source: Path, out: Path, case: dict, plan_sha: str, qualification_sha: str
) -> None:
    details = json.loads((source / "case.json").read_bytes())
    former_job = str(details["slurm_job_id"])
    evidence = verify_case_outputs(
        source, case, plan_sha, former_job, qualification_sha
    )
    source_case = source / "case"
    if not mesh_check_passed((source_case / "log.checkMesh").read_text()):
        raise ValueError("retained matched case failed the mesh check")
    convergence = read_convergence(
        (source_case / MONITOR_PATH).read_text(),
        (source_case / "log.simpleFoam").read_text(),
        case["iteration_limit"],
    )
    if convergence != evidence["convergence"]:
        raise ValueError("retained convergence evidence differs from the solver logs")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("log.checkMesh", "log.simpleFoam", MONITOR_PATH):
        target = out / "case" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_case / name, target)
    shutil.copytree(source / "native", out / "native")
    _write_json(
        out / "case.json",
        details
        | {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "recovered_from": {
                "job_id": former_job,
                "completion": evidence["completion"],
            },
        },
    )
    completion = json.loads((source / "completion.json").read_bytes())
    _write_json(
        out / "completion.json",
        completion | {"case_sha256": _sha256(out / "case.json")},
    )


def run_case(
    plan: Path,
    case_id: str,
    generator_repo: Path,
    out: Path,
    *,
    qualification: Path | None = None,
    reference_mesh_exception: bool = False,
    completed_output: Path | None = None,
) -> None:
    if os.environ.get("WM_PROJECT_VERSION") not in ("2112", "v2112"):
        raise RuntimeError("source the OpenFOAM v2112 runtime before running a case")
    if os.environ.get("SLURM_NTASKS") != str(CORES):
        raise RuntimeError("run the CFD case in a SLURM allocation with 16 tasks")
    out = out.resolve()
    raw_plan = plan.read_bytes()
    spec = json.loads(raw_plan)
    if spec["schema"] != 1 or spec["generator_revision"] != GENERATOR_REVISION:
        raise ValueError("unexpected case manifest or generator revision")
    matches = [case for case in spec["cases"] if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case ID must occur exactly once in the manifest")
    case = matches[0]
    plan_sha256 = hashlib.sha256(raw_plan).hexdigest()
    reference_mesh = None
    if reference_mesh_exception:
        if "group" in case:
            raise ValueError("reference mesh exception cannot apply to matched cases")
        reference_mesh = approved_reference_mesh(case_id, plan_sha256)
    qualification_sha256 = None
    if "group" in case:
        if qualification is None:
            raise ValueError("matched CFD requires three qualifying reproductions")
        gate = check_qualification(qualification)
        if gate["official_split_sha256"] != spec["sources"]["airfrans_split_sha256"]:
            raise ValueError(
                "qualification and matched case use different official splits"
            )
        qualification_sha256 = _sha256(qualification)
    if completed_output is not None:
        if qualification_sha256 is None:
            raise ValueError("only matched CFD can recover a completed export")
        _retain_completed_case(
            completed_output, out, case, plan_sha256, qualification_sha256
        )
        return
    out.mkdir(parents=True, exist_ok=False)
    source = out / "generator"
    generator_sha256 = _generator_archive(generator_repo, source)
    _write_json(
        out / "case.json",
        {
            "plan_sha256": plan_sha256,
            "generator_revision": GENERATOR_REVISION,
            "generator_archive_sha256": generator_sha256,
            "openfoam_version": os.environ["WM_PROJECT_VERSION"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cores": CORES,
            "case": case,
            "qualification_sha256": qualification_sha256,
        }
        | ({"reference_mesh_exception": reference_mesh} if reference_mesh else {}),
    )
    case_dir = out / "case"
    _setup_case(source, case, plan.resolve().parent, case_dir)
    internal, wall, convergence = solve_case(
        case_dir, case["iteration_limit"], reference_mesh=reference_mesh
    )
    convert_files(internal, wall, out / "native")
    _write_json(
        out / "completion.json",
        {
            "case": case_id,
            "convergence": convergence,
            "case_sha256": _sha256(out / "case.json"),
            "mesh_log_sha256": _sha256(case_dir / "log.checkMesh"),
            "solver_log_sha256": _sha256(case_dir / "log.simpleFoam"),
            "force_monitor_sha256": _sha256(case_dir / MONITOR_PATH),
            "conversion_sha256": _sha256(out / "native/conversion.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("generator_repo", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--reference-mesh-exception", action="store_true")
    parser.add_argument("--completed-output", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_case(
        args.plan,
        args.case_id,
        args.generator_repo,
        args.out,
        qualification=args.qualification,
        reference_mesh_exception=args.reference_mesh_exception,
        completed_output=args.completed_output,
    )
    print(args.out)


if __name__ == "__main__":
    main()
