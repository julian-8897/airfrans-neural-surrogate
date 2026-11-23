import gzip
import hashlib
import json

import pytest

from airfoil_surrogate.cfd import reference_mesh
from airfoil_surrogate.cfd.accounting import Allocation, budget
from airfoil_surrogate.cfd.campaign import matched_timeout_retries


def test_reference_exception_binds_plan_case_mesh_and_the_single_failure(
    tmp_path, monkeypatch
):
    mesh = tmp_path / "points.gz"
    mesh.write_bytes(gzip.compress(b"audited mesh"))
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "plan_sha256": "original-plan",
                "cases": {
                    "reference": {
                        "max_aspect_ratio": "1200.5",
                        "flagged_cells": 4,
                        "poly_mesh_sha256": {
                            mesh.name: hashlib.sha256(b"audited mesh").hexdigest()
                        },
                    }
                },
            }
        )
    )
    monkeypatch.setattr(reference_mesh, "PIN", pin)
    for case, plan in (("matched", "original-plan"), ("reference", "replaced-plan")):
        with pytest.raises(ValueError, match="outside the approved cases"):
            reference_mesh.approved_reference_mesh(case, plan)

    evidence = reference_mesh.approved_reference_mesh("reference", "original-plan")
    log = (
        " ***High aspect ratio cells found, Max aspect ratio: 1200.5, "
        "number of cells 4\n\nFailed 1 mesh checks.\n"
    )
    reference_mesh.verify_reference_mesh(log, tmp_path, evidence)
    for invalid in (
        log.replace("Failed 1", "Failed 2"),
        log.replace("cells 4", "cells 5"),
    ):
        with pytest.raises(ValueError, match="only the pinned high aspect ratio"):
            reference_mesh.verify_reference_mesh(invalid, tmp_path, evidence)
    mesh.write_bytes(gzip.compress(b"changed mesh"))
    with pytest.raises(ValueError, match="differs from the audited mesh"):
        reference_mesh.verify_reference_mesh(log, tmp_path, evidence)


def test_requalification_reserves_three_sixty_minute_allocations_explicitly():
    proposed = {"reproduction": 3, "matched": 0}
    assert budget({}, {}, proposed)["proposed_core_seconds"] == 40 * 3600
    requalification = budget({}, {}, proposed, reference_requalification=True)
    assert requalification["proposed_core_seconds"] == 48 * 3600


def test_revised_cpu_cap_requires_an_explicit_argument():
    proposed = {"matched": 72}
    assert not budget({}, {}, proposed)["within_cap"]
    raised = budget({}, {}, proposed, cap_core_seconds=675 * 3600)
    assert raised["within_cap"]
    assert raised["cap_core_seconds"] == 675 * 3600


def test_longer_matched_wall_time_requires_a_same_case_timeout():
    allocations = {
        "1": Allocation(16, 16, 2110, 35, "TIMEOUT", "0:0"),
        "2": Allocation(0, 16, 0, 45, "PENDING", "0:0"),
    }
    first = dict(phase="matched", job_id="1", case="uiuc", outcome="retry")
    retry = dict(phase="matched", job_id="2", case="uiuc", outcome="pending")
    phases = {"1": "matched", "2": "matched"}
    with pytest.raises(ValueError, match="no same-case allocation timeout"):
        matched_timeout_retries([first, retry], allocations)
    with pytest.raises(ValueError, match="exceeds its phase's wall-time limit"):
        budget(allocations, phases, {})
    first["reason"] = "allocation_timeout"
    permitted = matched_timeout_retries([first, retry], allocations)
    recorded = budget(allocations, phases, {}, matched_recovery_jobs=permitted)
    assert recorded["charged_core_seconds"] == 16 * (2110 + 45 * 60)
