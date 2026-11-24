"""Qualify the reproductions and freeze the CFD-qualified matched cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from airfoil_surrogate.cfd.accounting import (
    TERMINAL,
    TOTAL_CORE_SECONDS,
    WALL_LIMIT_MINUTES,
    Allocation,
    budget,
    read_allocations,
)
from airfoil_surrogate.cfd.reproduction import GENERATOR_REVISION
from airfoil_surrogate.eval.forces import AIRFRANS_VISCOSITY

QUOTAS = {("uiuc", None): 30, ("naca", "naca4"): 15, ("naca", "naca5"): 15}
SCIENTIFIC_FAILURES = {"geometry", "mesh", "nonfinite_output", "force_convergence"}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _write(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_ledger(path: Path, allocations: dict[str, Allocation]) -> list[dict]:
    attempts = []
    for entry in _json(path)["attempts"]:
        exp_path = (path.parent / entry["exp_run"]).resolve()
        exp = _json(exp_path)
        if exp["backend"]["kind"] != "slurm":
            raise ValueError("expected a registered Slurm attempt")
        job = str(exp["backend"]["job_id"])
        if job not in allocations:
            raise ValueError(f"missing allocation record for {job}")
        attempt = dict(
            entry,
            job_id=job,
            run_id=exp["run_id"],
            exp_status=exp["status"],
            exp_run=_record(exp_path),
        )
        for key in ("cfd_dir", "verification", "evidence"):
            if key in entry:
                attempt[key] = str((path.parent / entry[key]).resolve())
        attempts.append(attempt)
    if len({a["job_id"] for a in attempts}) != len(attempts) or len(
        {a["run_id"] for a in attempts}
    ) != len(attempts):
        raise ValueError("duplicate CFD attempt in the execution ledger")
    return attempts


def _terminal_attempts(
    attempts: list[dict], allocations: dict[str, Allocation]
) -> dict[tuple[str, str], list[dict]]:
    by_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in attempts:
        allocation = allocations[row["job_id"]]
        if allocation.state not in TERMINAL:
            raise ValueError(f"job {row['job_id']} is not terminal")
        outcome = row["outcome"]
        if outcome == "qualified":
            if (allocation.state, allocation.exit_code, row["exp_status"]) != (
                "COMPLETED",
                "0:0",
                "succeeded",
            ):
                raise ValueError(
                    "qualified attempt did not complete successfully in Slurm and exp"
                )
        elif outcome in ("retry", "reject"):
            if not row.get("reason") or not row.get("evidence"):
                raise ValueError("failed attempts require a reason and evidence file")
            if outcome == "reject" and row["reason"] not in SCIENTIFIC_FAILURES:
                raise ValueError(
                    "replacement requires a prespecified scientific failure"
                )
            if outcome == "retry" and row["reason"] in SCIENTIFIC_FAILURES:
                raise ValueError(
                    "scientific rejections cannot be retried as infrastructure"
                )
        else:
            raise ValueError("expected a recorded qualified, retry or reject outcome")
        records = by_case[row["phase"], row["case"]]
        if records and records[-1]["outcome"] != "retry":
            raise ValueError(
                "only an infrastructure retry may be followed by another attempt"
            )
        evidence = (
            {"evidence": _record(Path(row["evidence"]))}
            if outcome != "qualified"
            else {}
        )
        records.append(
            row
            | evidence
            | {"allocation": asdict(allocation), "core_seconds": allocation.charge}
        )
    return by_case


def matched_timeout_retries(
    attempts: list[dict], allocations: dict[str, Allocation]
) -> frozenset[str]:
    previous = {}
    approved = set()
    for row in attempts:
        if row["phase"] != "matched":
            continue
        job, case = row["job_id"], row["case"]
        if allocations[job].limit_minutes > WALL_LIMIT_MINUTES["matched"]:
            prior = previous.get(case)
            if (
                prior is None
                or prior["outcome"] != "retry"
                or prior.get("reason") != "allocation_timeout"
                or allocations[prior["job_id"]].state != "TIMEOUT"
            ):
                raise ValueError(f"{job} has no same-case allocation timeout to retry")
            approved.add(job)
        previous[case] = row
    return frozenset(approved)


def final_matched_recovery(
    attempts: list[dict], allocations: dict[str, Allocation]
) -> str:
    case_attempts = [
        row
        for row in attempts
        if (row["phase"], row["case"]) == ("matched", "uiuc-s3016")
    ]
    longer = [
        row["job_id"]
        for row in case_attempts
        if allocations[row["job_id"]].limit_minutes > 45
    ]
    if len(longer) != 1:
        raise ValueError("expected one approved final recovery for uiuc-s3016")
    index = next(i for i, row in enumerate(case_attempts) if row["job_id"] == longer[0])
    if index == 0 or case_attempts[index - 1]["job_id"] != "928510":
        raise ValueError("the final recovery must directly follow timeout 928510")
    return longer[0]


def verify_case_outputs(
    root: Path,
    case: dict,
    plan_sha: str,
    job_id: str,
    qualification_sha: str | None = None,
) -> dict:
    details = _json(root / "case.json")
    completion = _json(root / "completion.json")
    if (details["case"], details["plan_sha256"], str(details["slurm_job_id"])) != (
        case,
        plan_sha,
        job_id,
    ):
        raise ValueError(f"CFD case provenance mismatch for {case['id']}")
    if details.get("qualification_sha256") != qualification_sha:
        raise ValueError(f"CFD qualification mismatch for {case['id']}")
    reference_mesh = details.get("reference_mesh_exception")
    if reference_mesh is not None and qualification_sha is not None:
        raise ValueError(f"reference mesh exception on a matched case: {case['id']}")
    if completion["case"] != case["id"] or not completion["convergence"]["passed"]:
        raise ValueError(f"CFD case has no passing completion for {case['id']}")
    paths = {
        "case_sha256": root / "case.json",
        "mesh_log_sha256": root / "case/log.checkMesh",
        "solver_log_sha256": root / "case/log.simpleFoam",
        "force_monitor_sha256": root
        / "case/postProcessing/forceCoeffs1/0/coefficient.dat",
        "conversion_sha256": root / "native/conversion.json",
    }
    for key, path in paths.items():
        if completion[key] != _sha(path):
            raise ValueError(f"changed CFD evidence for {case['id']}: {key}")
    outputs = _json(paths["conversion_sha256"])["output"]
    native = {}
    for name in ("internal.vtu", "aerofoil.vtp"):
        path = root / "native" / name
        if (
            path.stat().st_size != outputs[name]["size"]
            or _sha(path) != outputs[name]["sha256"]
        ):
            raise ValueError(f"changed native mesh for {case['id']}: {name}")
        native[name] = {
            "path": str(path.resolve()),
            "size": outputs[name]["size"],
            "sha256": outputs[name]["sha256"],
        }
    return {
        "completion": _record(root / "completion.json"),
        "native": native,
        "convergence": completion["convergence"],
    } | ({"reference_mesh_exception": reference_mesh} if reference_mesh else {})


def _verify_case(
    row: dict, case: dict, plan_sha: str, qualification_sha: str | None = None
) -> dict:
    verified = verify_case_outputs(
        Path(row["cfd_dir"]), case, plan_sha, row["job_id"], qualification_sha
    )
    return (
        case | verified | {key: row[key] for key in ("job_id", "run_id", "allocation")}
    )


def qualify_reproductions(plan: Path, by_case: dict) -> dict:
    spec = _json(plan)
    plan_sha = _sha(plan)
    if (
        spec["generator_revision"] != GENERATOR_REVISION
        or len({c["id"] for c in spec["cases"]}) != 3
        or len(spec["cases"]) != 3
    ):
        raise ValueError("expected the three pinned reproduction cases")
    if {case for phase, case in by_case if phase == "reproduction"} != {
        c["id"] for c in spec["cases"]
    }:
        raise ValueError("reproduction attempts do not match the case plan")
    if any(phase != "reproduction" for phase, _ in by_case):
        raise ValueError("record qualification before any matched CFD allocations")
    cases = []
    for case in spec["cases"]:
        row = by_case["reproduction", case["id"]][-1]
        if row["outcome"] != "qualified":
            raise ValueError(f"reproduction has no qualified attempt for {case['id']}")
        verified = _verify_case(row, case, plan_sha)
        evidence = Path(row["verification"])
        report = _json(evidence)
        if (
            report["case"] != case["id"]
            or report["plan_sha256"] != plan_sha
            or not report["passed"]
            or report.get("reference_mesh_exception")
            != verified.get("reference_mesh_exception")
            or report["internal_sha256"] != verified["native"]["internal.vtu"]["sha256"]
            or report["aerofoil_sha256"] != verified["native"]["aerofoil.vtp"]["sha256"]
        ):
            raise ValueError(f"reproduction comparison did not qualify {case['id']}")
        cases.append(verified | {"verification": _record(evidence)})
    return {
        "schema": 1,
        "passed": True,
        "reproduction_plan": _record(plan),
        "official_split_sha256": spec["official_split_sha256"],
        "cases": cases,
    }


def retained_cases(
    cohorts: list[dict], by_case: dict, quotas: dict = QUOTAS
) -> list[tuple[dict, dict]]:
    """Take the approved prefix of each group, skipping documented rejections."""
    retained = []
    considered = set()
    matched_attempted = {case for phase, case in by_case if phase == "matched"}
    for (group, family), count in quotas.items():
        candidates = sorted(
            (
                case
                for case in cohorts
                if (case["group"], case.get("family")) == (group, family)
            ),
            key=lambda case: case["rank"],
        )
        if [case["rank"] for case in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError("non-contiguous candidate ranking")
        group_retained = []
        for case in candidates:
            records = by_case.get(("matched", case["id"]), [])
            if not records or records[-1]["outcome"] == "retry":
                raise ValueError(
                    f"matched cohort incomplete at rank {case['rank']} "
                    f"for {group}/{family}"
                )
            considered.add(case["id"])
            if records[-1]["outcome"] == "qualified":
                group_retained.append((case, records[-1]))
            if len(group_retained) == count:
                break
        if len(group_retained) != count:
            raise ValueError(
                f"matched cohort has {len(group_retained)}/{count} cases "
                f"for {group}/{family}"
            )
        retained.extend(group_retained)
    if considered != matched_attempted:
        raise ValueError("attempted a case outside the required ranked prefix")
    return retained


def check_qualification(qualification: Path) -> dict:
    gate = _json(qualification)
    if (
        gate.get("schema") != 1
        or gate.get("passed") is not True
        or len(gate["cases"]) != 3
        or len({item["id"] for item in gate["cases"]}) != 3
    ):
        raise ValueError("three passing reproductions must qualify the CFD pipeline")
    for case in gate["cases"]:
        verification = case["verification"]
        raw = Path(verification["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != verification["sha256"]:
            raise ValueError("a reproduction verification report changed")
        result = json.loads(raw)
        if result.get("passed") is not True or result.get("case") != case["id"]:
            raise ValueError("qualification contains a failing reproduction report")
    return gate


def freeze_cohorts(cohorts: Path, qualification: Path, by_case: dict) -> dict:
    gate = check_qualification(qualification)
    if {case for phase, case in by_case if phase == "reproduction"} != {
        item["id"] for item in gate["cases"]
    }:
        raise ValueError("reproduction allocations differ from the qualification")
    for case in gate["cases"]:
        records = by_case.get(("reproduction", case["id"]), [])
        if (
            not records
            or records[-1]["job_id"] != case["job_id"]
            or records[-1]["outcome"] != "qualified"
        ):
            raise ValueError(
                "reproduction allocation is missing from the campaign ledger"
            )
    spec = _json(cohorts)
    plan_sha = _sha(cohorts)
    if spec["generator_revision"] != GENERATOR_REVISION or len(
        {case["id"] for case in spec["cases"]}
    ) != len(spec["cases"]):
        raise ValueError("unknown generator or duplicate matched case ID")
    if gate["official_split_sha256"] != spec["sources"]["airfrans_split_sha256"]:
        raise ValueError(
            "matched cohort and reproduction used different AirfRANS splits"
        )
    retained = retained_cases(spec["cases"], by_case)
    conditions = {
        (
            case["inlet_velocity"],
            case["reynolds"],
            case["aoa_deg"],
            case["iteration_limit"],
        )
        for case, _ in retained
    }
    if conditions != {(4e6 * AIRFRANS_VISCOSITY, 4e6, 4.0, 20_000)}:
        raise ValueError("matched cohorts must share the proposed operating point")
    qualification_record = _record(qualification)
    return {
        "schema": 1,
        "qualification": qualification_record,
        "candidate_manifest": _record(cohorts),
        "cases": [
            _verify_case(row, case, plan_sha, qualification_record["sha256"])
            for case, row in retained
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("budget")
    estimate.add_argument("--reproduction", type=int, default=0)
    estimate.add_argument("--matched", type=int, default=0)
    estimate.add_argument("--qualification", type=Path)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("reproduction_plan", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("cohorts", type=Path)
    freeze.add_argument("qualification", type=Path)
    for action in (estimate, qualify, freeze):
        action.add_argument("ledger", type=Path)
        action.add_argument("accounting", type=Path)
        action.add_argument("--preparation-accounting", type=Path)
        action.add_argument("--reference-requalification", action="store_true")
        action.add_argument("--matched-timeout-recovery", action="store_true")
        action.add_argument("--final-matched-recovery", action="store_true")
        action.add_argument(
            "--cpu-core-hours", type=int, default=TOTAL_CORE_SECONDS // 3600
        )
        action.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    allocations = read_allocations(args.accounting.read_text())
    attempts = read_ledger(args.ledger, allocations)
    phases = {row["job_id"]: row["phase"] for row in attempts}
    proposals = (
        {phase: getattr(args, phase) for phase in ("reproduction", "matched")}
        if args.command == "budget"
        else {}
    )
    preparation = (
        read_allocations(args.preparation_accounting.read_text(), required_cpus=None)
        if args.preparation_accounting is not None
        else None
    )
    accounting = budget(
        allocations,
        phases,
        proposals,
        preparation=preparation,
        reference_requalification=args.reference_requalification,
        cap_core_seconds=args.cpu_core_hours * 3600,
        matched_recovery_jobs=(
            matched_timeout_retries(attempts, allocations)
            if args.matched_timeout_recovery
            else frozenset()
        ),
        final_matched_recovery_job=(
            final_matched_recovery(attempts, allocations)
            if args.final_matched_recovery
            else None
        ),
    ) | {
        "ledger": _record(args.ledger),
        "sacct": _record(args.accounting),
    }
    if args.preparation_accounting is not None:
        accounting["preparation_sacct"] = _record(args.preparation_accounting)
    if args.command == "budget":
        if args.matched:
            if args.qualification is None:
                raise ValueError(
                    "matched allocations require three qualifying reproductions"
                )
            gate = check_qualification(args.qualification)
            if any(
                phases.get(case["job_id"]) != "reproduction"
                or allocations[case["job_id"]].state != "COMPLETED"
                for case in gate["cases"]
            ):
                raise ValueError("a reproduction allocation is missing or incomplete")
            accounting["qualification"] = _record(args.qualification)
        _write(args.out, accounting)
        if not accounting["within_cap"]:
            raise SystemExit(1)
        return
    if not accounting["within_cap"]:
        raise ValueError("CFD allocations exceeded the campaign cap")
    by_case = _terminal_attempts(attempts, allocations)
    result = (
        qualify_reproductions(args.reproduction_plan, by_case)
        if args.command == "qualify"
        else freeze_cohorts(args.cohorts, args.qualification, by_case)
    )
    _write(
        args.out,
        result
        | {
            "budget": accounting,
            "attempts": [row for records in by_case.values() for row in records],
        },
    )


if __name__ == "__main__":
    main()
