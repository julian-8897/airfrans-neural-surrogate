import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

runner = pytest.importorskip("airfoil_surrogate.cfd.run_case")

CONTROL = """writeInterval $endTime;
functions
{
    otherObject
    {
        writeInterval $endTime;
    }
    forceCoeffs1
    {
        writeFields yes;
        writeControl timeStep;
        writeInterval $endTime;
        /*binData
        {
            nBin 20;
        }*/
    }
}
"""


def test_force_monitor_keeps_global_and_other_output_cadences():
    expected = CONTROL.replace("writeFields yes;", "writeFields no;").replace(
        "writeControl timeStep;\n        writeInterval $endTime;",
        "writeControl timeStep;\n        writeInterval 1;",
    )
    assert runner.per_iteration_monitor(CONTROL) == expected
    with pytest.raises(ValueError, match="expected one writeFields"):
        runner.per_iteration_monitor(CONTROL.replace("writeFields yes;", ""))


@pytest.mark.parametrize("external_shape", [False, True])
def test_run_case_uses_one_preparation_and_solver_path_for_both_geometry_sources(
    external_shape, tmp_path, monkeypatch
):
    geometry = np.array([[1, 0], [0.5, 0.1], [0, 0], [0.5, -0.1], [1, 0]])
    case = {
        "id": "test",
        "inlet_velocity": 62.0,
        "aoa_deg": 4,
        "iteration_limit": 20_000,
    }
    if external_shape:
        path = tmp_path / "shape.csv"
        np.savetxt(path, geometry, delimiter=",")
        case |= {
            "geometry_file": path.name,
            "geometry_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    else:
        case["naca_parameters"] = [2.0, 4.0, 12.0]

    plan = tmp_path / "cases.json"
    plan.write_text(
        json.dumps(
            {
                "schema": 1,
                "generator_revision": runner.GENERATOR_REVISION,
                "cases": [case],
            }
        )
    )
    out = tmp_path / "run"
    source = out / "generator"
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2112")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    def fake_archive(_, path):
        (path / runner.TEMPLATE / "system").mkdir(parents=True)
        return "pinned-archive"

    calls = []

    def naca_generator(parameters, **kwargs):
        assert parameters == [2.0, 4.0, 12.0]
        assert kwargs["CTE"] and kwargs["nb_samples"] == 2000
        calls.append("naca_generator")
        return geometry

    def dict_grading(*args):
        np.testing.assert_array_equal(args[-1], geometry)
        assert args[-2] == 4
        calls.append("dict_grading")
        return "grading"

    def block_mesh(path, points, grading):
        assert Path(path) == out / "case"
        np.testing.assert_array_equal(points, geometry)
        assert grading == "grading"
        calls.append("blockMeshDict_generator")

    def system(path, inlet, angle, ranks, **kwargs):
        assert (inlet, angle, ranks) == (62.0, 4, 16)
        assert kwargs == {"n_iter": 20_000, "compressible": False}
        (Path(path) / "system/controlDict").write_text(CONTROL)
        calls.append("system_generator")

    generator = SimpleNamespace(
        __file__=str(source / "simulation_generator.py"),
        params=dict(
            zip(
                ("L", "y_h", "y_hd", "x_h", "y_exp", "x_exp", "x_expd"),
                (200, 2e-6, 1e-4, 1e-5, 1.075, 1.025, 1.075),
                strict=True,
            )
        ),
        naca_generator=naca_generator,
        dict_grading=dict_grading,
        blockMeshDict_generator=block_mesh,
        system_generator=system,
        init_generator=lambda *_: calls.append("init_generator"),
    )
    monkeypatch.setattr(runner, "_generator_archive", fake_archive)
    monkeypatch.setattr(runner.importlib, "import_module", lambda _: generator)

    def command(path, log, argv):
        calls.append(argv[0])
        if log == "log.checkMesh":
            (path / log).write_text("Mesh OK.\n")
        elif log == "log.simpleFoam":
            assert argv == ["mpirun", "-np", "16", "simpleFoam", "-parallel"]
            (path / log).write_text("Time = 1000\nEnd\n")
            monitor = path / runner.MONITOR_PATH
            monitor.parent.mkdir(parents=True)
            monitor.write_text(
                "# Time Cd Cs Cl\n"
                + "".join(f"{i} 0.01 0 0.5\n" for i in range(1, 1001))
            )
        elif log == "log.foamToVTK":
            leaf = path / "VTK/final"
            (leaf / "boundary").mkdir(parents=True)
            (leaf / "internal.vtu").touch()
            (leaf / "boundary/aerofoil.vtp").touch()

    def convert(internal, wall, destination):
        assert internal == out / "case/VTK/final/internal.vtu" and internal.exists()
        assert wall == out / "case/VTK/final/boundary/aerofoil.vtp" and wall.exists()
        assert destination == out / "native"
        destination.mkdir()
        (destination / "conversion.json").write_text("{}")
        calls.append("convert")

    monkeypatch.setattr(runner, "_command", command)
    monkeypatch.setattr(runner, "convert_files", convert)
    runner.run_case(plan, "test", tmp_path, out)
    assert calls == ([] if external_shape else ["naca_generator"]) + [
        "dict_grading",
        "blockMeshDict_generator",
        "system_generator",
        "init_generator",
        "blockMesh",
        "checkMesh",
        "decomposePar",
        "mpirun",
        "reconstructPar",
        "foamToVTK",
        "convert",
    ]
    completion = json.loads((out / "completion.json").read_text())
    assert completion["convergence"]["passed"] and completion["case"] == "test"
    assert completion["conversion_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert (
        out / "case/system/controlDict"
    ).read_text() == runner.per_iteration_monitor(CONTROL)
    with pytest.raises(FileExistsError):
        runner.run_case(plan, "test", tmp_path, out)


def test_case_runner_stops_after_failed_mesh_check(tmp_path, monkeypatch):
    calls = []

    def command(path, log, argv):
        calls.append(argv[0])
        if log == "log.checkMesh":
            (path / log).write_text("Failed 1 mesh checks.\n")

    monkeypatch.setattr(runner, "_command", command)
    with pytest.raises(ValueError, match="checkMesh failed"):
        runner.solve_case(tmp_path, 20_000)
    assert calls == ["blockMesh", "checkMesh"]
