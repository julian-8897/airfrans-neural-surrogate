import hashlib
import json
from pathlib import Path

import pytest

from airfoil_surrogate.data.staged_original import case_order_sha256

report = pytest.importorskip("airfoil_surrogate.eval.study_report")


def test_assemble_reproduces_reports_from_registered_cases_and_rejects_changed_evidence(
    tmp_path,
    monkeypatch,
):
    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def reference(path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def registered(phase, job):
        experiment = "synthetic-study"
        name = f"{phase}-{job}"
        return write(
            f"records/{name}/run.json",
            {
                "experiment": experiment,
                "run_id": name,
                "status": "succeeded",
                "backend": {"kind": "slurm", "job_id": str(job)},
                "output_dir": f"outputs/{name}",
            },
        )

    runs = {
        role: registered(role, job) for job, role in enumerate(sorted(report.RUN_ROLES))
    }
    output = {
        role: json.loads(path.read_text())["output_dir"] for role, path in runs.items()
    }

    data_pin = {"zip_index_sha256": "zip", "split_sha256": "splits"}
    ids = {
        "train": [f"train-{i}" for i in range(12)],
        "validation": ["validation"],
        "test": ["test 0", "test 1"],
    }
    pin = data_pin | {
        "partitions": {
            part: {"size": len(names), "order_sha256": case_order_sha256(names)}
            for part, names in ids.items()
        }
    }

    def stage(part, **fields):
        cases = [
            {"id": name, "files": {"internal.vtu": {"sha256": name}}}
            for name in ids[part]
        ]
        return write(
            f"{part}/stage.json",
            {
                "schema": 1,
                "source": data_pin,
                "partition": part,
                "partition_size": len(cases),
                "cases": cases,
                **fields,
            },
        ), cases

    train_stage, _ = stage("train")
    val_stage, _ = stage("validation")
    train_sha = reference(train_stage)["sha256"]
    val_sha = reference(val_stage)["sha256"]
    audit = write(
        f"{output['force_audit']}/force-audit.json",
        {
            "passed": True,
            "cases": [
                {
                    "forces": {
                        "adapter_minus_direct": [[0, 0, 0], [0, 0, 0]],
                        "direct_minus_stored": [1e-6, 1e-7],
                    }
                }
                for _ in range(12)
            ],
        },
    )
    qualification = output["qualification"]
    parity = write(
        f"{qualification}/transolver-parity.json",
        {
            "passed": True,
            "checks": {"full_mesh": {"max_abs_by_channel": [1e-6] * 4}},
        },
    )
    preflight = write(
        f"{qualification}/training-preflight/preflight.json",
        {
            "projection": {"within_training_reservation": True},
        },
    )
    training = f"{output['training']}/training"
    frozen_hashes = {}
    for filename, value in {
        "run.json": {
            "inputs": {
                "train_stage_sha256": train_sha,
                "validation_stage_sha256": val_sha,
                "qualification": {
                    name + "_sha256": reference(path)["sha256"]
                    for name, path in (
                        ("force_audit", audit),
                        ("parity", parity),
                        ("preflight", preflight),
                    )
                },
            },
            "config": {"seed": 0},
        },
        "model.pt": {},
        "normaliser.json": {},
        "selection.json": {},
        "metrics.jsonl": {},
    }.items():
        frozen_hashes[filename] = reference(write(f"{training}/{filename}", value))[
            "sha256"
        ]
    frozen_hashes["baselines"] = reference(
        write(f"{output['baselines']}/baselines.json", {})
    )["sha256"]
    frozen = write(
        "frozen.json",
        {
            "schema": 1,
            "training_stage_sha256": train_sha,
            "selected": {"epoch": 210, "score": 0.2},
            "sha256": frozen_hashes,
        },
    )
    frozen_sha = reference(frozen)["sha256"]
    test_stage, official_cases = stage("test", frozen_sha256=frozen_sha)

    attempts = []

    def cfd_case(phase, name, job):
        path = registered(phase, job)
        run_id = json.loads(path.read_text())["run_id"]
        attempts.append(
            {
                "exp_run": reference(path),
                "exp_status": "succeeded",
                "phase": phase,
                "outcome": "qualified",
                "run_id": run_id,
                "case": name,
                "job_id": str(job),
                "core_seconds": 160,
            }
        )
        return {"id": name, "run_id": run_id, "job_id": str(job)}

    reproductions = []
    for i in range(3):
        case = cfd_case("reproduction", f"repro-{i}", 100 + i)
        verification = write(
            f"verify/{case['id']}/reproduction.json",
            {
                "passed": True,
                "case": case["id"],
                "train_stage_sha256": train_sha,
                "coefficient_absolute_difference": [0.01, 0.0001],
                "cp_rmse": [0.02, 0.03],
            },
        )
        write(f"verify/{case['id']}/surface-cp.png", {})
        write(f"verify/{case['id']}/surface-cp.csv", {})
        reproductions.append(case | {"verification": reference(verification)})
    gate = write(
        "qualified.json",
        {
            "schema": 1,
            "passed": True,
            "cases": reproductions,
            "official_split_sha256": pin["split_sha256"],
        },
    )
    sources = {
        "airfrans_split_sha256": pin["split_sha256"],
        "uiuc_index": reference(write("uiuc-index.html", {})),
        "uiuc_coordinates": reference(write("uiuc-coordinates.zip", {})),
    }
    candidate = write("candidate.json", {"sources": sources})
    write("screening.json", {"sources": sources})
    from airfoil_surrogate.eval.forces import AIRFRANS_VISCOSITY

    matched = []
    for group, family, count in (
        ("uiuc", None, 30),
        ("naca", "naca4", 15),
        ("naca", "naca5", 15),
    ):
        for i in range(count):
            matched.append(
                cfd_case("matched", f"{family or group}-{i}", 200 + len(matched))
                | {
                    "group": group,
                    "family": family,
                    "inlet_velocity": 4e6 * AIRFRANS_VISCOSITY,
                    "reynolds": 4e6,
                    "aoa_deg": 4.0,
                    "iteration_limit": 20_000,
                    "native": {"internal.vtu": {"sha256": f"{group}-{i}"}},
                }
            )
    cohorts = write(
        "cohorts.json",
        {
            "schema": 1,
            "qualification": reference(gate),
            "candidate_manifest": reference(candidate),
            "cases": matched,
            "attempts": attempts,
            "budget": {
                "within_cap": True,
                "charged_core_seconds": 160 * len(attempts),
                "cap_core_seconds": 600 * 3600,
                "proposed_core_seconds": 0,
                "sacct": reference(write("cfd-accounting.json", {})),
                "ledger": reference(write("cfd-ledger.json", {})),
            },
        },
    )
    gpu = write(
        "gpu.json",
        {
            "experiment": "synthetic-study",
            "phase": None,
            "within_cap": True,
            "charged_gpu_seconds": 240,
            "cap_gpu_seconds": 24 * 3600,
            "proposed_gpu_seconds": 0,
            "reserved_future_gpu_seconds": 0,
            "unsettled_jobs": [],
            "accounting": reference(write("gpu-accounting.json", {})),
            "attempts": [
                {
                    "charged_gpu_seconds": 60,
                    "phase": role,
                    "status": "succeeded",
                    "exp_run": reference(runs[role]),
                    "run_id": json.loads(runs[role].read_text())["run_id"],
                    "job_id": json.loads(runs[role].read_text())["backend"]["job_id"],
                }
                for role in report.PHASE_LIMIT_MINUTES
            ],
        },
    )

    def write_rows(directory, cases, transfer):
        rows = []
        for i, case in enumerate(cases):
            name = case["id"]
            tag = i / 100
            coeffs = [[1.0 + tag, 0, 0], [0.1 + tag, 0, 0]]
            metric = {
                key: [0.5] * (2 if key.startswith("coefficient_") else 4)
                for key in (
                    "volume_mse",
                    "surface_mse",
                    "volume_relative_l2",
                    "surface_relative_l2",
                    "coefficient_absolute_error",
                    "coefficient_relative_error",
                    "volume_area_weighted_mse",
                    "surface_arc_length_weighted_mse",
                )
            }
            path = f"predictions/{name}.npz"
            prediction_sha = reference(write(f"{directory}/{path}", {"i": i}))["sha256"]
            row = {
                "case": name,
                "reference_coefficients": coeffs,
                "transolver": metric
                | {
                    "coefficients": coeffs,
                    "raw_coefficients": coeffs,
                    "wall_pressure_rmse": 1.0,
                    "wall_shear_rmse": [1.0, 1.0],
                },
            }
            if transfer:
                row |= {
                    "group": case["group"],
                    "family": case["family"],
                    "mesh_files": case["native"],
                    "cfd_job_id": case["job_id"],
                    "cfd_run_id": case["run_id"],
                    "prediction": {"path": path, "sha256": prediction_sha},
                }
            else:
                row |= {
                    "mesh_files": case["files"],
                    "prediction_file": path,
                    "prediction_sha256": prediction_sha,
                    "baseline": {
                        key: [x + 0.5 for x in value] for key, value in metric.items()
                    }
                    | {"coefficients": [coeffs[0][0], coeffs[1][0]]},
                }
            rows.append(row)
        (tmp_path / directory / "cases.jsonl").write_text(
            "\n".join(map(json.dumps, rows)) + "\n"
        )
        return rows

    official_dir = f"{output['official']}/test"
    transfer_dir = f"{output['transfer']}/transfer"
    for directory, metadata in (
        (
            official_dir,
            {"split": "test", "test_stage_sha256": reference(test_stage)["sha256"]},
        ),
        (
            transfer_dir,
            {
                "split": "matched_transfer",
                "cohort_sha256": reference(cohorts)["sha256"],
            },
        ),
    ):
        write(
            f"{directory}/evaluation.json",
            metadata
            | {
                "frozen_sha256": frozen_sha,
                "selected_checkpoint": frozen_hashes["model.pt"],
            },
        )
    official = report.summarise_official(
        write_rows(official_dir, official_cases, False)
    )
    transfer, rankings = report.summarise_transfer(
        write_rows(transfer_dir, matched, True)
    )
    write(f"{official_dir}/summary.json", official)
    write(f"{transfer_dir}/report/summary.json", transfer)
    write(f"{transfer_dir}/report/rankings.json", rankings)

    spec = write(
        "report-inputs.json",
        {
            "experiment": "synthetic-study",
            "runs": {
                role: str(path.relative_to(tmp_path)) for role, path in runs.items()
            },
            "train_stage": "train",
            "validation_stage": "validation",
            "test_stage": "test",
            "freeze": frozen.name,
            "cohorts": cohorts.name,
            "gpu_budget": gpu.name,
            "protocol": "docs/methodology/original-mesh-transfer-protocol.md",
        },
    )
    write("docs/methodology/original-mesh-transfer-protocol.md", {})
    out = tmp_path / "report"
    report.assemble_report(spec, pin, tmp_path, out)

    text = (out / "report.md").read_text()
    assert "30 UIUC sections" in text and "epoch 210" in text
    assert "| CL absolute error | 0.5 | 1 | -0.5 [-0.5, -0.5] | yes |" in text
    assert "[Cp profiles](../verify/repro-0/surface-cp.png)" in text
    assert f"[exp {runs['official'].parent.name}]" in text
    evidence = json.loads((out / "evidence.json").read_text())
    assert evidence["artifacts"]["checkpoint"]["sha256"] == frozen_hashes["model.pt"]
    assert len(evidence["cfd_reproductions"]) == 3

    from airfoil_surrogate.eval import official_report

    study = "synthetic-study"
    roles = {
        role: json.loads(runs[role].read_text())
        for role in official_report.ROLES
        if role in runs
    }
    roles["official_stage"] = json.loads(registered("official-stage", 501).read_text())
    roles["official_stage"]["output_dir"] = "."
    gate_record = json.loads(registered("reproduction-gate", 502).read_text())
    gate_out = gate_record["output_dir"]
    write(f"{gate_out}/reproduction-gate.md", "Two mesh failures and one timeout")
    write(
        f"{gate_out}/metrics.json",
        {
            "reproduction_gate_passed": False,
            "mesh_failures": 2,
            "solver_timeouts": 1,
        },
    )
    cpu_jobs = [record["backend"]["job_id"] for record in roles.values()]
    gpu_jobs = [
        roles[key]["backend"]["job_id"]
        for key in ("qualification", "training", "official")
    ]
    roles["reproduction_gate"] = gate_record
    records = {}
    for role, record in roles.items():
        run_id = record["run_id"]
        write(f"experiments/{study}/runs/{run_id}/run.json", record)
        records[role] = {"run_id": run_id, "job_id": record["backend"]["job_id"]}
    partial = write(
        "official-report-inputs.json",
        {
            "experiment": study,
            "runs": records,
            "cpu_jobs": cpu_jobs,
            "gpu_jobs": gpu_jobs,
            "report_reserved_core_seconds": 600,
        },
    )
    monkeypatch.setattr(official_report, "PIN", write("pin.json", pin))

    def accounting(ids, fields, out):
        if "AllocTRES" in fields:
            result = "".join(
                f"{job}|gres/gpu:a100=1|gres/gpu:a100=1|15|30|COMPLETED|0:0\n"
                for job in ids
            )
        else:
            result = "".join(f"{job}|2|2|15|30|COMPLETED|0:0\n" for job in ids)
        out.write_text(result)
        return result

    monkeypatch.setattr(official_report, "_sacct", accounting)
    partial_out = tmp_path / "official-report"
    official_report.assemble(partial, tmp_path, partial_out)
    partial_text = (partial_out / "report.md").read_text()
    assert "The approved reproduction gate failed. 2 mesh checks failed" in partial_text
    assert "30 UIUC" not in partial_text
    assert "| CL absolute error | 0.5 | 1 | -0.5 [-0.5, -0.5] | yes |" in partial_text

    from airfoil_surrogate.eval import transfer_report

    reference_study = "synthetic-requalification"
    write(
        f"experiments/{study}/experiment.json",
        {"id": study, "parent": reference_study},
    )
    inference = json.loads(runs["transfer"].read_text())
    inference_id = inference["run_id"]
    write(f"experiments/{study}/runs/{inference_id}/run.json", inference)
    split_cohorts = json.loads(cohorts.read_text())
    for attempt in split_cohorts["attempts"]:
        if attempt["phase"] == "reproduction":
            source = json.loads(Path(attempt["exp_run"]["path"]).read_bytes())
            linked = write(
                f"requalification/{source['run_id']}/run.json",
                source | {"experiment": reference_study},
            )
            attempt["exp_run"] = reference(linked)
    requalified = write("requalified-cohorts.json", split_cohorts)
    transfer_metadata = f"{transfer_dir}/evaluation.json"
    original_metadata = json.loads((tmp_path / transfer_metadata).read_text())
    write(
        transfer_metadata,
        original_metadata
        | {
            "cohort_sha256": reference(requalified)["sha256"],
            "cases": [[case["id"], case["group"]] for case in matched],
            "operating_point": {
                key: matched[0][key]
                for key in ("inlet_velocity", "aoa_deg", "reynolds")
            },
        },
    )
    matched_spec = write(
        "matched-report-inputs.json",
        {
            "experiment": study,
            "transfer_run_id": inference_id,
            "training": str(tmp_path / training),
            "baselines": str(tmp_path / output["baselines"] / "baselines.json"),
            "freeze": str(frozen),
            "cohorts": str(requalified),
        },
    )
    monkeypatch.setattr(transfer_report, "PIN", write("pin.json", pin))
    matched_out = tmp_path / "matched-report"
    transfer_report.assemble(matched_spec, tmp_path, matched_out)
    matched_result = json.loads((matched_out / "evidence.json").read_text())
    assert matched_result["case_count"] == 60
    assert len(matched_result["reproduction_reports"]) == 3
    write(transfer_metadata, original_metadata)
    with pytest.raises(ValueError, match="provenance differs"):
        transfer_report.assemble(matched_spec, tmp_path, tmp_path / "wrong-matched")

    invalid_out = tmp_path / "rejected"
    training_record = runs["training"].read_text()
    runs["training"].write_text(
        json.dumps(json.loads(training_record) | {"status": "failed"})
    )
    with pytest.raises(ValueError, match="training requires a successful run"):
        report.assemble_report(spec, pin, tmp_path, invalid_out)
    assert not invalid_out.exists()
    runs["training"].write_text(training_record)

    write(f"{official_dir}/summary.json", official | {"case_count": 200})
    with pytest.raises(ValueError, match="saved summary differs"):
        report.assemble_report(spec, pin, tmp_path, invalid_out)
    assert not invalid_out.exists()
    write(f"{official_dir}/summary.json", official)
    write(f"{official_dir}/predictions/test 0.npz", {"changed": True})
    with pytest.raises(ValueError, match="missing or changed prediction"):
        report.assemble_report(spec, pin, tmp_path, invalid_out)
    assert not invalid_out.exists()
