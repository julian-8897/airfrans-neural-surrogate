import json

import pytest

from airfoil_surrogate.gpu_budget import (
    GPU_CAP_SECONDS,
    gpu_budget,
    read_gpu_allocations,
    read_gpu_ledger,
)

A100 = "billing=117,cpu=2,gres/gpu:a100=1,gres/gpu=1,mem=12G,node=1"


def test_gpu_budget_charges_terminal_use_and_full_live_wall_limit():
    rows = read_gpu_allocations(
        f"10|{A100}|{A100}|1200|30|COMPLETED|0:0\n"
        f"11||{A100}|0|1380|PENDING|0:0\n"
        f"12||{A100}|0|30|CANCELLED by 600|0:0\n"
    )
    entries = {
        "10": {"phase": "qualification", "status": "succeeded"},
        "11": {"phase": "training", "status": "running"},
        "12": {"phase": "official", "status": "killed"},
    }
    assert rows["10"].charge == 1200
    assert rows["11"].charge == 1380 * 60
    assert rows["12"].charge == 0
    result = gpu_budget(entries, rows, phase="official", minutes=15, reserve_minutes=15)
    assert result["charged_gpu_seconds"] == 84_000
    assert result["within_cap"]
    assert result["unsettled_jobs"] == ["11"]
    assert not result["can_submit"]
    queued = gpu_budget(
        entries, rows, phase="official", minutes=30, after_training="11"
    )
    assert queued["can_submit"] and queued["after_training"] == "11"
    assert queued["proposed_gpu_seconds"] == 1800
    assert not gpu_budget(
        entries, rows, phase="official", minutes=30, after_training="10"
    )["can_submit"]
    with pytest.raises(ValueError, match="only to official"):
        gpu_budget(entries, rows, phase="transfer", minutes=30, after_training="11")

    with pytest.raises(ValueError, match="registered job IDs differ"):
        gpu_budget(entries, {key: rows[key] for key in ("10", "11")})


def test_gpu_budget_enforces_prerequisites_cap_and_single_retained_evaluation():
    rows = read_gpu_allocations(
        f"10|{A100}|{A100}|1800|30|COMPLETED|0:0\n"
        f"11|{A100}|{A100}|34200|1380|NODE_FAIL|1:0\n"
    )
    ledger = {
        "10": {"phase": "qualification", "status": "succeeded"},
        "11": {"phase": "training", "status": "failed"},
    }
    result = gpu_budget(ledger, rows, phase="training", minutes=810)
    assert result["reserved_future_gpu_seconds"] == 1800
    assert (
        result["charged_gpu_seconds"] + result["proposed_gpu_seconds"] + 1800
        == GPU_CAP_SECONDS
    )
    assert result["can_submit"]
    assert not gpu_budget(ledger, rows, phase="training", minutes=811)["within_cap"]
    assert not gpu_budget(ledger, rows, phase="official", minutes=1)["can_submit"]
    assert not gpu_budget(ledger, rows, phase="qualification", minutes=1)["can_submit"]

    rows |= read_gpu_allocations(
        f"12|{A100}|{A100}|46200|810|COMPLETED|0:0\n"
        f"13|{A100}|{A100}|420|15|COMPLETED|0:0\n"
    )
    ledger |= {
        "12": {"phase": "training", "status": "succeeded"},
        "13": {"phase": "official", "status": "succeeded"},
    }
    assert not gpu_budget(ledger, rows, phase="official", minutes=1)["can_submit"]
    assert gpu_budget(ledger, rows, phase="transfer", minutes=15)["can_submit"]
    ledger["13"]["status"] = "failed"
    with pytest.raises(ValueError, match="Slurm and exp disagree"):
        gpu_budget(ledger, rows, phase="official", minutes=1)


def test_gpu_accounting_rejects_ambiguous_and_non_full_a100_allocations():
    complete = f"10|{A100}|{A100}|20|30|COMPLETED|0:0\n"
    with pytest.raises(ValueError, match="ambiguous GPU allocation"):
        read_gpu_allocations(complete + complete)
    with pytest.raises(ValueError, match="ambiguous GPU allocation"):
        read_gpu_allocations(complete.replace("10|", "10.batch|"))
    with pytest.raises(ValueError, match="single full A100"):
        read_gpu_allocations(complete.replace("gpu:a100", "gpu:a100-mig"))
    with pytest.raises(ValueError, match="invalid full-A100 allocation"):
        read_gpu_allocations(f"10||{A100}|20|30|COMPLETED|0:0")


def test_gpu_ledger_binds_live_and_closed_attempts_to_exp(tmp_path):
    experiment = "original-study"
    run_file = tmp_path / "run.json"
    run = {
        "run_id": "timing-id",
        "experiment": experiment,
        "status": "running",
        "backend": {"kind": "slurm", "job_id": None},
    }
    run_file.write_text(json.dumps(run))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "phase": "qualification",
                        "job_id": "123",
                        "exp_run": run_file.name,
                    }
                ]
            }
        )
    )
    assert read_gpu_ledger(ledger, experiment)["123"]["run_id"] == "timing-id"
    run["backend"]["job_id"] = "456"
    run["status"] = "succeeded"
    run_file.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="disagrees with its exp job ID"):
        read_gpu_ledger(ledger, experiment)
