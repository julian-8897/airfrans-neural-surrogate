import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from meshes import grid_with_hole
from numpy.testing import assert_allclose, assert_array_equal

from airfoil_surrogate.data.features import NodeFeatures, Normaliser
from airfoil_surrogate.data.staged_original import case_order_sha256
from airfoil_surrogate.models.transolver import Transolver, TransolverConfig

pv = pytest.importorskip("pyvista")
original = pytest.importorskip("airfoil_surrogate.data.original")


@pytest.fixture
def mesh_files(tmp_path):
    pos, triangles, node_id = grid_with_hole({(2, 2)})
    points = np.column_stack((pos, np.full(len(pos), 0.5)))
    cells = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    internal = pv.UnstructuredGrid(
        cells, np.full(len(triangles), pv.CellType.TRIANGLE, dtype=np.uint8), points
    )
    wall = [node_id(3, 3), node_id(3, 2), node_id(2, 2), node_id(2, 3)]
    velocity = np.tile([2.0, 0.5, 0.0], (len(pos), 1))
    velocity[[0, *wall]] = 0.0
    internal.point_data["U"] = velocity
    internal.cell_data["U"] = np.tile([-99.0, 0.0, 0.0], (len(triangles), 1))
    internal.point_data["p"] = pos[:, 0]
    internal.point_data["nut"] = np.full(len(pos), 1e-5)
    internal.point_data["implicit_distance"] = -np.ones(len(pos))
    internal.point_data["implicit_distance"][wall] = 0.0

    patch = pv.PolyData(points[wall], lines=[2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
    outward = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]])
    patch.point_data["Normals"] = -outward / np.sqrt(2)
    patch.cell_data["Normals"] = np.array(
        [[-1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [0, -1.0, 0]]
    )
    internal_path = tmp_path / "internal.vtu"
    patch_path = tmp_path / "aerofoil.vtp"
    internal.save(internal_path)
    patch.save(patch_path)
    return internal_path, patch_path, internal, patch, wall, outward


def test_original_features_preserve_mesh_and_derive_wall_from_patch(mesh_files):
    internal_path, patch_path, internal, _, wall, outward = mesh_files
    mesh = original.load_original_mesh(internal_path, patch_path)
    features = mesh.features(inlet_velocity=4.0, aoa=np.pi / 2)

    assert_array_equal(mesh.internal.cells, internal.cells)
    assert_array_equal(mesh.wall_nodes, wall)
    assert_array_equal(features.pos, internal.points[:, :2])
    assert_array_equal(features.x[:, :2], features.pos)
    assert_allclose(
        features.x[:, 2:4], np.tile([0, 4], (len(features.pos), 1)), atol=1e-15
    )
    assert_array_equal(features.x[:, 4], (~features.surf).astype(float))
    assert_array_equal(np.flatnonzero(features.surf), sorted(wall))
    assert_array_equal(features.x[~features.surf, 5:7], 0)
    assert_allclose(features.x[wall, 5:7], outward[:, :2] / np.sqrt(2))
    assert_array_equal(features.y[:, :2], internal.point_data["U"][:, :2])
    assert_array_equal(features.y[:, 2], internal.point_data["p"])
    assert_array_equal(features.y[:, 3], internal.point_data["nut"])
    assert features.x.dtype == features.y.dtype == np.float64


def test_original_mesh_rejects_a_mismatched_wall_patch(mesh_files):
    internal_path, patch_path, _, patch, _, _ = mesh_files
    patch.points[0, 0] += 1e-6
    patch.save(patch_path)
    with pytest.raises(ValueError, match="wall patch must match"):
        original.load_original_mesh(internal_path, patch_path)


def test_original_input_qualification_uses_the_adjacent_fluid_cells(mesh_files):
    audit = pytest.importorskip("airfoil_surrogate.eval.qualify_forces")
    mesh = original.load_original_mesh(*mesh_files[:2])
    sim = audit.OriginalForceSimulation(mesh, inlet_velocity=2.0, aoa=0.0)
    passed = audit.input_checks(mesh, sim)
    assert passed["passed"] and passed["wall_nodes"] == 4
    assert passed["min_segment_alignment"] > 0
    assert passed["min_point_alignment"] > 0
    mesh.aerofoil.cell_data["Normals"] *= -1
    failed = audit.input_checks(mesh, sim)
    assert not failed["passed"] and failed["min_segment_alignment"] < 0
    assert failed["max_normal_length_error"] == passed["max_normal_length_error"]


def test_force_qualification_checks_components_and_the_two_stored_totals():
    audit = pytest.importorskip("airfoil_surrogate.eval.qualify_forces")
    reference = np.array([[0.5, 0.4, 0.1], [0.04, 0.03, 0.01]])
    stored = reference[:, 0].copy()
    assert audit.compare_forces(reference, reference, stored)["passed"]

    adapter = reference.copy()
    adapter[1, 1:] += [2e-8, -2e-8]
    result = audit.compare_forces(adapter, reference, stored)
    assert not result["adapter_passed"] and result["stored_label_passed"]
    assert_allclose(result["adapter_minus_direct"][1], [0.0, 2e-8, -2e-8])

    stored[0] -= 2e-5
    result = audit.compare_forces(reference, reference, stored)
    assert result["adapter_passed"] and not result["stored_label_passed"]
    assert_allclose(result["direct_minus_stored"], [2e-5, 0.0])


def test_original_force_adapter_on_known_fields(mesh_files):
    forces = pytest.importorskip("airfoil_surrogate.eval.forces")
    internal_path, patch_path, _, _, _, outward = mesh_files
    mesh = original.load_original_mesh(internal_path, patch_path)
    sim = forces.OriginalForceSimulation(mesh, inlet_velocity=2.0, aoa=0.0)
    reference = forces.force_coefficients(sim, reference=True)

    fields = np.zeros((len(sim.position), 4))
    fields[:, 2] = 17.0
    sim.set_fields(fields)
    assert_allclose(forces.force_coefficients(sim), 0, atol=1e-12)

    # AirfRANS averages the unit diagonal vertex normals onto each edge. Their
    # edge-average magnitudes are 1/sqrt(2), scaling the exact rectangular-wall
    # force of -0.5 for dp/dx = 1. The dynamic pressure at U = 2 is 2.
    fields[:, 2] += sim.position[:, 0]
    sim.set_fields(fields)
    drag = -0.25 / np.sqrt(2)
    assert_allclose(
        forces.force_coefficients(sim), [[0, 0, 0], [drag, drag, 0]], rtol=0, atol=1e-12
    )

    fields[:, 0] = sim.position[:, 1]
    sim.set_fields(fields)
    assert_allclose(
        sim.wallshearstress(over_airfoil=True),
        sim.NU * outward[:, [1, 0]] / np.sqrt(2),
        rtol=0,
        atol=np.finfo(np.float64).eps * sim.NU,
    )
    assert_array_equal(forces.force_coefficients(sim, reference=True), reference)


@pytest.fixture
def training_stage(mesh_files, tmp_path):
    ids = ["airFoil2D_SST_32_-4_1_4_12", "airFoil2D_SST_54_3_2_3_1_14"]
    pin = {
        "split_sha256": "a" * 64,
        "zip_index_sha256": "b" * 64,
        "partitions": {"train": {"size": 2, "order_sha256": case_order_sha256(ids)}},
    }
    root = tmp_path / "train"
    cases = []
    for case in ids:
        (root / case).mkdir(parents=True)
        files = {}
        for source in mesh_files[:2]:
            contents = source.read_bytes()
            (root / case / f"{case}_{source.name}").write_bytes(contents)
            files[source.name] = {
                "size": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        cases.append({"id": case, "files": files})
    manifest = root / "stage.json"
    metadata = {
        "schema": 1,
        "source": pin,
        "partition": "train",
        "partition_size": 2,
        "cases": cases,
    }
    manifest.write_text(json.dumps(metadata))
    return root, pin, metadata


def test_normaliser_fit_verifies_training_membership_and_staged_meshes(
    training_stage, tmp_path
):
    fit = pytest.importorskip("airfoil_surrogate.data.fit_original_normaliser")
    root, pin, metadata = training_stage
    manifest = root / "stage.json"
    cases = metadata["cases"]
    ids = [row["id"] for row in cases]
    output = tmp_path / "normaliser.json"
    fit.fit_staged_training(root, pin, output)

    saved = json.loads(output.read_text())
    assert saved["fit_cases"] == ids
    assert (
        saved["fit_stage_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    normaliser = Normaliser.load(output)
    angles = np.deg2rad([-4.0, 3.0])
    inlet = np.array([32.0, 54.0]) * np.exp(1j * angles)
    expected = inlet.mean()
    assert_allclose(normaliser.mean_in[2:4], [expected.real, expected.imag], atol=1e-12)
    with pytest.raises(FileExistsError):
        fit.fit_staged_training(root, pin, output)

    wrong_id = {**cases[1], "id": "airFoil2D_SST_54_3_1_4_17"}
    manifest.write_text(json.dumps({**metadata, "cases": [cases[0], wrong_id]}))
    retry = tmp_path / "retry.json"
    with pytest.raises(ValueError, match="full official train order"):
        fit.fit_staged_training(root, pin, retry)

    manifest.write_text(json.dumps(metadata))
    path = root / ids[1] / f"{ids[1]}_internal.vtu"
    contents = path.read_bytes()
    path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        fit.fit_staged_training(root, pin, retry)
    assert not retry.exists()


def test_baseline_fit_reads_verified_fields_and_reference_forces(
    training_stage, mesh_files, tmp_path
):
    module = pytest.importorskip("airfoil_surrogate.eval.original_baselines")
    root, pin, metadata = training_stage
    output = tmp_path / "baselines.json"
    module.fit_staged_baselines(root, pin, output)

    record = json.loads(output.read_text())
    ids = [row["id"] for row in metadata["cases"]]
    assert record["fit_cases"] == ids
    assert (
        record["fit_stage_sha256"]
        == hashlib.sha256((root / "stage.json").read_bytes()).hexdigest()
    )
    baseline = module.OriginalBaselines.from_dict(record)
    internal, patch = mesh_files[:2]
    mesh = original.load_original_mesh(internal, patch)
    for case, (inlet, theta) in zip(ids, [(32, -4), (54, 3)], strict=True):
        sim = module.OriginalForceSimulation(mesh, inlet, np.deg2rad(theta))
        reference = module.force_coefficients(sim, reference=True)[:, 0]
        assert_allclose(baseline.predict_coefficients(case), reference, rtol=0, atol=0)
    fields = sim.features
    expected = np.stack((fields.y[~fields.surf].mean(0), fields.y[fields.surf].mean(0)))
    assert_allclose(baseline.regional_means, expected, rtol=0, atol=0)


def test_verified_normaliser_uses_population_statistics_over_all_nodes():
    fit = pytest.importorskip("airfoil_surrogate.data.fit_original_normaliser")
    cases = []
    for values in ([1.0, 3.0], [8.0]):
        nodes = np.array(values)[:, None]
        cases.append(
            NodeFeatures(
                pos=np.tile(nodes, (1, 2)),
                x=np.tile(nodes, (1, 7)),
                y=np.tile(2 * nodes, (1, 4)),
                surf=np.zeros(len(nodes), dtype=bool),
            )
        )
    norm = fit.fit_checked_normaliser(cases)
    # The three equally weighted input nodes have mean 4 and variance 26/3.
    assert_allclose(norm.mean_in, 4, rtol=0, atol=1e-15)
    assert_allclose(norm.std_in, np.sqrt(26 / 3), rtol=0, atol=1e-15)
    assert_allclose(norm.mean_out, 8, rtol=0, atol=1e-15)
    assert_allclose(norm.std_out, 2 * np.sqrt(26 / 3), rtol=0, atol=1e-15)


def test_frozen_evaluation_uses_each_native_mesh_once_and_rebuilds_summary(
    training_stage, mesh_files, tmp_path, monkeypatch
):
    fit = pytest.importorskip("airfoil_surrogate.data.fit_original_normaliser")
    baselines = pytest.importorskip("airfoil_surrogate.eval.original_baselines")
    freeze = pytest.importorskip("airfoil_surrogate.eval.freeze_original")
    evaluation = pytest.importorskip("airfoil_surrogate.eval.original")
    from airfoil_surrogate.train_original import OriginalTrainConfig

    train_root, pin, train = training_stage

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    test_ids = ["airFoil2D_SST_35_2_1_4_12", "airFoil2D_SST_67_8_2_3_1_14"]
    pin["partitions"]["test"] = {"size": 2, "order_sha256": case_order_sha256(test_ids)}
    run = tmp_path / "run"
    run.mkdir()
    norm_file = run / "normaliser.json"
    fit.fit_staged_training(train_root, pin, norm_file)
    baseline_file = tmp_path / "baselines.json"
    baselines.fit_staged_baselines(train_root, pin, baseline_file)
    cfg = TransolverConfig(n_hidden=8, n_layers=1, n_head=2, slice_num=4, ref=2)
    training_cfg = OriginalTrainConfig(epochs=1, val_every=1, model=cfg)
    torch.manual_seed(0)
    model = Transolver(cfg)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, run / "model.pt")
    (run / "run.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "train_stage_sha256": sha(train_root / "stage.json"),
                    "qualification": {
                        "parity_sha256": "parity",
                        "preflight_sha256": "preflight",
                        "force_audit_sha256": "force-audit",
                    },
                },
                "normaliser_sha256": sha(norm_file),
                "config": asdict(training_cfg),
            }
        )
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "validation_score": 1.0}) + "\n"
    )
    (run / "selection.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "score": 1.0,
                "checkpoint_sha256": sha(run / "model.pt"),
                "validation_scores": [{"epoch": 1, "score": 1.0}],
            }
        )
    )
    frozen = tmp_path / "frozen.json"
    freeze.freeze_evaluation(run, baseline_file, pin, frozen, config=training_cfg)

    test_root = tmp_path / "test"
    test_rows = []
    for case_id, source in zip(test_ids, train["cases"], strict=True):
        (test_root / case_id).mkdir(parents=True)
        source_dir = train_root / source["id"]
        for suffix in source["files"]:
            old = source_dir / f"{source['id']}_{suffix}"
            (test_root / case_id / f"{case_id}_{suffix}").write_bytes(old.read_bytes())
        test_rows.append({"id": case_id, "files": source["files"]})
    (test_root / "stage.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "source": pin,
                "partition": "test",
                "partition_size": 2,
                "cases": test_rows,
                "frozen_sha256": sha(frozen),
            }
        )
    )
    calls = []
    forward = evaluation.Transolver.forward

    def counted(self, x, pos):
        calls.append((x.shape, pos.shape))
        return forward(self, x, pos)

    monkeypatch.setattr(evaluation.Transolver, "forward", counted)
    device = torch.device("cpu")
    out = tmp_path / "evaluation"
    evaluation.evaluate_official(
        run, baseline_file, frozen, test_root, pin, out, device
    )
    rows = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert [row["case"] for row in rows] == test_ids
    assert all(row["nodes"] == mesh_files[2].n_points for row in rows)
    assert calls == [((1, row["nodes"], 7), (1, row["nodes"], 2)) for row in rows]
    for case, row in zip(test_rows, rows, strict=True):
        path = out / row["prediction_file"]
        assert sha(path) == row["prediction_sha256"]
        assert row["mesh_files"] == case["files"]
        with np.load(path) as saved:
            assert saved["fields"].shape == (row["nodes"], 4)
        assert (
            row["transolver"]["raw_coefficients"] != row["transolver"]["coefficients"]
        )
        assert row["transolver"]["surface_relative_l2"][:2] == [None, None]
    fresh = tmp_path / "rebuilt.json"
    evaluation.rebuild_summary(out, fresh)
    assert fresh.read_bytes() == (out / "summary.json").read_bytes()
    assert len(calls) == 2

    baseline_file.write_bytes(baseline_file.read_bytes() + b"\n")
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="frozen evaluation input changed: baselines"):
        evaluation.evaluate_official(
            run, baseline_file, frozen, test_root, pin, rejected, device
        )
    assert not rejected.exists() and len(calls) == 2


def test_paired_intervals_preserve_pairing_and_undefined_channels():
    evaluation = pytest.importorskip("airfoil_surrogate.eval.original")
    delta = np.array([[-1.0, np.nan, 1.0], [-1.0, np.nan, 3.0]])
    combinations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    first, undefined, third = evaluation.paired_mean_differences(delta, combinations)
    assert first == {
        "mean": -1.0,
        "interval95": [-1.0, -1.0],
        "n_cases": 2,
        "n_draws": 4,
        "supports_improvement": True,
    }
    assert undefined == {
        "mean": None,
        "interval95": None,
        "n_cases": 0,
        "n_draws": 0,
        "supports_improvement": False,
    }
    assert_allclose(third["interval95"], [1.075, 2.925])


def _cfd_history(last_cl, last_cd):
    samples = [
        f"{i} {0.01 if i <= 500 else last_cd} 0 {0.5 if i <= 500 else last_cl}\n"
        for i in range(1, 1001)
    ]
    return "# Time Cd Cs Cl\n" + "".join(samples)


def test_cfd_monitor_uses_consecutive_final_iterations_and_named_force_columns():
    cfd = pytest.importorskip("airfoil_surrogate.cfd.reproduction")
    solver_log = "Time = 1000\n\nEnd\n"
    history = _cfd_history(0.501, 0.01004)
    passed = cfd.read_convergence(history, solver_log, 20_000)
    assert passed["passed"]
    assert passed["last_iteration"] == 1000
    assert_allclose(passed["absolute_difference"], [0.001, 0.00004], atol=1e-15)
    assert not cfd.read_convergence(_cfd_history(0.501, 0.01006), solver_log, 20_000)[
        "passed"
    ]

    for bad_history in (
        history.replace("\n650 ", "\n649 "),
        history.removesuffix(history.splitlines()[-1] + "\n"),
    ):
        with pytest.raises(ValueError, match="final 1,000 consecutive iterations"):
            cfd.read_convergence(bad_history, solver_log, 20_000)
    with pytest.raises(ValueError, match="completed solve"):
        cfd.read_convergence(history, "Time = 1000\n", 20_000)


def test_cfd_reproduction_gate_checks_both_surfaces_independently(mesh_files):
    cfd = pytest.importorskip("airfoil_surrogate.cfd.reproduction")
    internal, patch = mesh_files[:2]
    mesh = original.load_original_mesh(internal, patch)
    case = {
        "id": "probe",
        "inlet_velocity": 2.0,
        "aoa_deg": 0,
        "iteration_limit": 20_000,
    }
    logs = ("Mesh OK.\n", "Time = 1000\n\nEnd\n", _cfd_history(0.5, 0.01))
    report, reference_cp, candidate_cp = cfd.compare_reproduction(
        case, mesh, mesh, *logs
    )
    assert report["passed"]
    assert reference_cp.shape == (2, 99)
    assert_allclose(
        reference_cp, np.tile(cfd.CP_STATIONS / 2, (2, 1)), rtol=0, atol=1e-12
    )
    assert_array_equal(reference_cp, candidate_cp)

    changed = mesh.internal.copy()
    changed.point_data["p"] += 0.2
    failed, *_ = cfd.compare_reproduction(
        case, mesh, replace(mesh, internal=changed), *logs
    )
    assert not failed["passed"]
    assert failed["mesh_ok"] and failed["convergence"]["passed"]
    assert_allclose(failed["coefficient_absolute_difference"], [0, 0], atol=1e-12)
    assert_allclose(failed["cp_rmse"], [0.1, 0.1], atol=1e-12)


def test_transfer_evaluation_freezes_full_cohorts_and_rebuilds_reports(
    tmp_path, mesh_files
):
    transfer = pytest.importorskip("airfoil_surrogate.eval.transfer")
    internal, wall, *_ = mesh_files
    norm = Normaliser(np.zeros(7), np.ones(7), np.zeros(4), np.ones(4))
    cfg = TransolverConfig(n_hidden=8, n_layers=1, n_head=2, slice_num=4, ref=2)
    torch.manual_seed(0)
    model = Transolver(cfg)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_json(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path

    torch.save(
        {"config": asdict(cfg), "model": model.state_dict()}, tmp_path / "model.pt"
    )
    write_json("normaliser.json", norm.to_dict())
    for name in ("run.json", "selection.json", "metrics.jsonl", "baselines"):
        write_json(name, {})
    inputs = {
        name: digest(tmp_path / name)
        for name in (
            "run.json",
            "selection.json",
            "metrics.jsonl",
            "normaliser.json",
            "model.pt",
            "baselines",
        )
    }
    frozen = write_json("frozen.json", {"schema": 1, "sha256": inputs})
    pin = {"split_sha256": "pinned-split"}
    gate = write_json(
        "gate.json", {"passed": True, "official_split_sha256": pin["split_sha256"]}
    )
    candidate = write_json(
        "candidate.json", {"sources": {"airfrans_split_sha256": pin["split_sha256"]}}
    )

    cases = []
    for (group, family), count in transfer.QUOTAS.items():
        for index in range(count):
            case_id = f"{family or group}-{index}"
            cases.append(
                {
                    "id": case_id,
                    "group": group,
                    "family": family,
                    "inlet_velocity": 4e6 * transfer.AIRFRANS_VISCOSITY,
                    "reynolds": 4e6,
                    "aoa_deg": 4.0,
                    "iteration_limit": 20_000,
                    "run_id": f"cfd-{case_id}",
                    "job_id": str(len(cases)),
                    "native": {
                        path.name: {
                            "path": str(path),
                            "sha256": digest(path),
                            "size": path.stat().st_size,
                        }
                        for path in (internal, wall)
                    },
                }
            )
    cohorts = {
        "schema": 1,
        "budget": {"within_cap": True},
        "qualification": {"path": str(gate), "sha256": digest(gate)},
        "candidate_manifest": {"path": str(candidate), "sha256": digest(candidate)},
        "cases": cases,
    }
    cohort_path = write_json("cohorts.json", cohorts)
    out = tmp_path / "output"
    device = torch.device("cpu")
    transfer.evaluate_transfer(
        tmp_path, tmp_path / "baselines", frozen, cohort_path, pin, out, device
    )

    entries = [
        json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()
    ]
    assert [row["case"] for row in entries] == [case["id"] for case in cases]
    assert entries[0]["cfd_run_id"] == "cfd-uiuc-0"
    pred = out / entries[0]["prediction"]["path"]
    assert digest(pred) == entries[0]["prediction"]["sha256"]
    with np.load(pred) as arrays:
        fields = arrays["fields"]
    assert fields.shape == (49, 4) and np.isfinite(fields).all()

    rebuilt = tmp_path / "rebuilt"
    transfer.rebuild_summary(out, rebuilt)
    for name in ("summary.json", "rankings.json"):
        assert (rebuilt / name).read_bytes() == (out / "report" / name).read_bytes()
    with pytest.raises(FileExistsError):
        transfer.evaluate_transfer(
            tmp_path, tmp_path / "baselines", frozen, cohort_path, pin, out, device
        )

    changed = tmp_path / "changed"
    cases[-1]["native"]["internal.vtu"]["sha256"] = "wrong"
    write_json("cohorts.json", cohorts)
    with pytest.raises(ValueError, match="frozen native mesh changed"):
        transfer.evaluate_transfer(
            tmp_path, tmp_path / "baselines", frozen, cohort_path, pin, changed, device
        )
    assert not changed.exists()

    cases.pop()
    write_json("cohorts.json", cohorts)
    with pytest.raises(ValueError, match="invalid frozen cohort membership"):
        transfer.evaluate_transfer(
            tmp_path, tmp_path / "baselines", frozen, cohort_path, pin, changed, device
        )
    assert not changed.exists()
