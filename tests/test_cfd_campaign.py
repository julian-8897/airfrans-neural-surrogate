import hashlib
import json

import pytest

from airfoil_surrogate.cfd.accounting import budget, read_allocations


def test_budget_charges_terminal_elapsed_and_live_requested_wall_time():
    rows = read_allocations(
        "10|16|16|1500|50|COMPLETED|0:0\n11|0|16|0|35|PENDING|0:0\n"
    )
    phases = {"10": "reproduction", "11": "matched"}
    assert budget(rows, phases, {"matched": 60}) == {
        "charged_core_seconds": 16 * (1500 + 35 * 60),
        "proposed_core_seconds": 60 * 16 * 35 * 60,
        "cap_core_seconds": 600 * 3600,
        "within_cap": True,
    }
    assert not budget(rows, phases, {"matched": 63})["within_cap"]
    with pytest.raises(ValueError, match="registered job IDs differ"):
        budget(rows, {"10": "reproduction"}, {})
    with pytest.raises(ValueError, match="ambiguous allocation record"):
        read_allocations("10.batch|16|16|1500|50|COMPLETED|0:0\n")


def test_reserve_rank_cannot_skip_an_unresolved_infrastructure_failure():
    campaign = pytest.importorskip("airfoil_surrogate.cfd.campaign")
    cohort = [{"id": str(rank), "group": "uiuc", "rank": rank} for rank in [3, 1, 2]]
    quotas = {("uiuc", None): 2}
    ledger = {
        ("matched", "1"): [{"outcome": "qualified"}],
        ("matched", "2"): [{"outcome": "retry"}],
        ("matched", "3"): [{"outcome": "qualified"}],
    }
    with pytest.raises(ValueError, match="incomplete at rank 2"):
        campaign.retained_cases(cohort, ledger, quotas)
    ledger["matched", "2"] = [{"outcome": "reject"}]
    cases = campaign.retained_cases(cohort, ledger, quotas)
    assert [case["id"] for case, _ in cases] == ["1", "3"]
    ledger["matched", "2"] = [{"outcome": "qualified"}]
    with pytest.raises(ValueError, match="outside the required ranked prefix"):
        campaign.retained_cases(cohort, ledger, quotas)


def test_freeze_binds_three_reproductions_and_sixty_terminal_case_outputs(tmp_path):
    campaign = pytest.importorskip("airfoil_surrogate.cfd.campaign")

    def write(path, data):
        path.write_text(json.dumps(data))
        return path

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_attempt(case, plan, job, phase, gate=None):
        root = tmp_path / f"run-{job}"
        monitor = root / "case/postProcessing/forceCoeffs1/0/coefficient.dat"
        monitor.parent.mkdir(parents=True)
        monitor.write_text("monitor")
        mesh_log = root / "case/log.checkMesh"
        mesh_log.write_text("mesh")
        solver_log = root / "case/log.simpleFoam"
        solver_log.write_text("solver")
        native = root / "native"
        native.mkdir()
        outputs = {}
        for name in ("internal.vtu", "aerofoil.vtp"):
            path = native / name
            path.write_bytes(name.encode())
            outputs[name] = {"sha256": digest(path), "size": len(name)}
        conversion = write(native / "conversion.json", {"output": outputs})
        case_file = write(
            root / "case.json",
            {
                "case": case,
                "plan_sha256": digest(plan),
                "slurm_job_id": job,
                "qualification_sha256": digest(gate) if gate else None,
            },
        )
        write(
            root / "completion.json",
            {
                "case": case["id"],
                "convergence": {"passed": True},
                "case_sha256": digest(case_file),
                "mesh_log_sha256": digest(mesh_log),
                "solver_log_sha256": digest(solver_log),
                "force_monitor_sha256": digest(monitor),
                "conversion_sha256": digest(conversion),
            },
        )
        record = {
            "phase": phase,
            "case": case["id"],
            "outcome": "qualified",
            "cfd_dir": str(root),
            "job_id": job,
            "run_id": f"exp-{job}",
            "allocation": {"elapsed_seconds": 100},
        }
        if phase == "reproduction":
            report = write(
                root / "reproduction.json",
                {
                    "case": case["id"],
                    "plan_sha256": digest(plan),
                    "passed": True,
                    "internal_sha256": outputs["internal.vtu"]["sha256"],
                    "aerofoil_sha256": outputs["aerofoil.vtp"]["sha256"],
                },
            )
            record["verification"] = str(report)
        return [record]

    reproduction = [{"id": f"repro-{i}"} for i in range(3)]
    plan = write(
        tmp_path / "reproduction-plan.json",
        {
            "generator_revision": campaign.GENERATOR_REVISION,
            "cases": reproduction,
            "official_split_sha256": "split",
        },
    )
    ledger = {
        ("reproduction", case["id"]): make_attempt(
            case, plan, str(i + 1), "reproduction"
        )
        for i, case in enumerate(reproduction)
    }
    gate = write(
        tmp_path / "qualified.json", campaign.qualify_reproductions(plan, ledger)
    )

    conditions = {
        "inlet_velocity": 4e6 * campaign.AIRFRANS_VISCOSITY,
        "reynolds": 4e6,
        "aoa_deg": 4.0,
        "iteration_limit": 20_000,
    }
    matched = []
    for (group, family), count in campaign.QUOTAS.items():
        for rank in range(1, count + 1):
            matched.append(
                {
                    "id": f"{group}-{family}-{rank}",
                    "group": group,
                    "family": family,
                    "rank": rank,
                    **conditions,
                }
            )
    cohorts = write(
        tmp_path / "cohorts.json",
        {
            "generator_revision": campaign.GENERATOR_REVISION,
            "sources": {"airfrans_split_sha256": "split"},
            "cases": matched,
        },
    )
    for i, case in enumerate(matched, start=10):
        ledger["matched", case["id"]] = make_attempt(
            case, cohorts, str(i), "matched", gate
        )

    final = campaign.freeze_cohorts(cohorts, gate, ledger)
    assert final["qualification"]["sha256"] == digest(gate)
    assert len(final["cases"]) == 60
    assert sum(row["group"] == "uiuc" for row in final["cases"]) == 30
    assert all(row["run_id"].startswith("exp-") for row in final["cases"])
    (tmp_path / "run-10/native/internal.vtu").write_bytes(b"changed mesh")
    with pytest.raises(ValueError, match="changed native mesh"):
        campaign.freeze_cohorts(cohorts, gate, ledger)
