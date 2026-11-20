"""Assemble the completed AirfRANS study from registered, saved outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from airfoil_surrogate.cfd.campaign import check_qualification
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import read_complete_partition
from airfoil_surrogate.eval.original import frozen_inputs, summarise_official
from airfoil_surrogate.eval.transfer import frozen_cases
from airfoil_surrogate.eval.transfer_metrics import summarise_transfer
from airfoil_surrogate.gpu_budget import PHASE_LIMIT_MINUTES

RUN_ROLES = {
    "force_audit",
    "qualification",
    "training",
    "baselines",
    "official",
    "transfer",
}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _checked_path(source: dict) -> Path:
    path = Path(source["path"])
    if _sha(path) != source["sha256"]:
        raise ValueError(f"registered evidence changed: {path}")
    return path


def _registered_runs(spec: dict, base: Path, root: Path) -> dict:
    if spec["runs"].keys() != RUN_ROLES:
        raise ValueError("provide one run record for every stage of the study")
    runs = {}
    for role, name in spec["runs"].items():
        path = (base / name).resolve()
        record = _json(path)
        if (
            record["experiment"] != spec["experiment"]
            or record["status"] != "succeeded"
        ):
            raise ValueError(
                f"{role} requires a successful run in the study experiment"
            )
        runs[role] = record | {
            "record": _source(path),
            "output": (root / record["output_dir"]).resolve(),
        }
    if len({run["run_id"] for run in runs.values()}) != len(runs):
        raise ValueError("study stages require distinct run records")
    return runs


def _checked_gpu_accounting(path: Path, experiment: str, runs: dict) -> dict:
    report = _json(path)
    _checked_path(report["accounting"])
    charged = sum(row["charged_gpu_seconds"] for row in report["attempts"])
    if (
        report["experiment"] != experiment
        or report["within_cap"] is not True
        or report["phase"] is not None
        or report["unsettled_jobs"]
        or report["proposed_gpu_seconds"] != 0
        or report["reserved_future_gpu_seconds"] != 0
        or charged != report["charged_gpu_seconds"]
        or charged > report["cap_gpu_seconds"]
    ):
        raise ValueError("expected terminal A100 accounting within the study cap")
    successful = {}
    for attempt in report["attempts"]:
        path = _checked_path(attempt["exp_run"])
        if attempt["status"] == "succeeded":
            phase = attempt["phase"]
            run = runs[phase]
            if (
                phase in successful
                or path.resolve() != Path(run["record"]["path"])
                or attempt["run_id"] != run["run_id"]
                or str(run["backend"]["job_id"]) != attempt["job_id"]
            ):
                raise ValueError("A100 accounting does not match the retained exp runs")
            successful[phase] = attempt["run_id"]
    if successful.keys() != PHASE_LIMIT_MINUTES.keys():
        raise ValueError("A100 accounting does not cover all retained Slurm phases")
    return report


def _cfd_reports(
    cohorts: dict, cases: list[dict], experiment: str, training_sha: str
) -> list[dict]:
    """Check links from each frozen CFD case to its registered attempt."""
    gate = check_qualification(_checked_path(cohorts["qualification"]))
    budget = cohorts["budget"]
    _checked_path(budget["sacct"])
    _checked_path(budget["ledger"])
    if "preparation_sacct" in budget:
        _checked_path(budget["preparation_sacct"])
    attempts = {}
    for row in cohorts["attempts"]:
        source = _checked_path(row["exp_run"])
        run = _json(source)
        if (
            run["experiment"] != experiment
            or run["run_id"] != row["run_id"]
            or str(run["backend"]["job_id"]) != row["job_id"]
            or row["job_id"] in attempts
        ):
            raise ValueError(
                "CFD allocation does not match a unique registered attempt"
            )
        attempts[row["job_id"]] = row
    if (
        budget["within_cap"] is not True
        or budget["proposed_core_seconds"] != 0
        or budget["charged_core_seconds"] + budget.get("preparation_core_seconds", 0)
        > budget["cap_core_seconds"]
        or budget["charged_core_seconds"]
        != sum(row["core_seconds"] for row in attempts.values())
    ):
        raise ValueError("expected terminal CFD accounting within the study cap")
    for phase, retained in (("reproduction", gate["cases"]), ("matched", cases)):
        for case in retained:
            row = attempts[case["job_id"]]
            if (
                row["phase"] != phase
                or row["case"] != case["id"]
                or row["outcome"] != "qualified"
                or row["exp_status"] != "succeeded"
                or row["run_id"] != case["run_id"]
            ):
                raise ValueError(f"unregistered qualified CFD case: {case['id']}")
    reproductions = []
    for case in gate["cases"]:
        verification = _checked_path(case["verification"])
        report = _json(verification)
        if report["train_stage_sha256"] != training_sha:
            raise ValueError(
                "CFD reproduction used a different original training stage"
            )
        reproductions.append(
            report
            | {
                "record": attempts[case["job_id"]]["exp_run"],
                "verification": case["verification"],
                "cp_plot": _source(verification.with_name("surface-cp.png")),
                "cp_data": _source(verification.with_name("surface-cp.csv")),
            }
        )
    return reproductions


def _prediction_rows(root: Path, cases: list[dict], *, transfer: bool) -> list[dict]:
    rows = [
        json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()
    ]
    if [row["case"] for row in rows] != [case["id"] for case in cases]:
        raise ValueError(f"incomplete or out-of-order predictions at {root}")
    for row, case in zip(rows, cases, strict=True):
        if transfer:
            prediction = row["prediction"]
            if (
                row["group"] != case["group"]
                or row["family"] != case.get("family")
                or row["cfd_run_id"] != case["run_id"]
                or row["cfd_job_id"] != case["job_id"]
                or row["mesh_files"] != case["native"]
            ):
                raise ValueError(
                    f"prediction does not reference the frozen CFD case: {row['case']}"
                )
        else:
            prediction = {
                "path": row["prediction_file"],
                "sha256": row["prediction_sha256"],
            }
            if row["mesh_files"] != case["files"]:
                raise ValueError(
                    f"prediction does not reference the official mesh: {row['case']}"
                )
        expected = (Path("predictions") / f"{case['id']}.npz").as_posix()
        if (
            prediction["path"] != expected
            or _sha(root / expected) != prediction["sha256"]
        ):
            raise ValueError(f"missing or changed prediction: {root / expected}")
    return rows


def _summary_matches(path: Path, regenerated: dict) -> None:
    if _json(path) != json.loads(json.dumps(regenerated, allow_nan=False)):
        raise ValueError(f"saved summary differs from the per-case metrics: {path}")


def assemble_report(spec_path: Path, pin: dict, root: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    spec = _json(spec_path)
    base = spec_path.parent
    runs = _registered_runs(spec, base, root)
    paths = {
        key: (base / spec[key]).resolve()
        for key in (
            "train_stage",
            "validation_stage",
            "test_stage",
            "freeze",
            "cohorts",
            "gpu_budget",
            "protocol",
        )
    }
    paths["force_audit"] = runs["force_audit"]["output"] / "force-audit.json"
    paths["parity"] = runs["qualification"]["output"] / "transolver-parity.json"
    paths["preflight"] = (
        runs["qualification"]["output"] / "training-preflight/preflight.json"
    )
    training = runs["training"]["output"] / "training"
    paths["checkpoint"] = training / "model.pt"
    paths["normaliser"] = training / "normaliser.json"
    paths["baselines"] = runs["baselines"]["output"] / "baselines.json"
    official_dir = runs["official"]["output"] / "test"
    transfer_dir = runs["transfer"]["output"] / "transfer"
    paths["official_summary"] = official_dir / "summary.json"
    paths["transfer_summary"] = transfer_dir / "report/summary.json"
    paths["rankings"] = transfer_dir / "report/rankings.json"

    frozen_sha = _sha(paths["freeze"])
    frozen = _json(paths["freeze"])
    inputs = frozen_inputs(training, paths["baselines"], frozen)
    run = json.loads(inputs["run.json"])
    training_rows, training_sha = read_complete_partition(
        paths["train_stage"], "train", pin
    )
    validation_rows, validation_sha = read_complete_partition(
        paths["validation_stage"], "validation", pin
    )
    test_rows, test_sha = read_complete_partition(
        paths["test_stage"], "test", pin, frozen_sha256=frozen_sha
    )
    audit = _json(paths["force_audit"])
    parity = _json(paths["parity"])
    preflight = _json(paths["preflight"])
    if (
        frozen["training_stage_sha256"] != training_sha
        or run["inputs"]["train_stage_sha256"] != training_sha
        or run["inputs"]["validation_stage_sha256"] != validation_sha
        or audit["passed"] is not True
        or len(audit["cases"]) != 12
        or parity["passed"] is not True
        or preflight["projection"]["within_training_reservation"] is not True
        or run["inputs"]["qualification"]
        != {
            name + "_sha256": _sha(paths[name])
            for name in ("parity", "preflight", "force_audit")
        }
    ):
        raise ValueError(
            "qualification evidence does not match the frozen training run"
        )

    cohorts = _json(paths["cohorts"])
    cases = frozen_cases(cohorts, pin)
    reproduction = _cfd_reports(cohorts, cases, spec["experiment"], training_sha)
    candidate = _checked_path(cohorts["candidate_manifest"])
    paths["candidate_manifest"] = candidate
    paths["screening"] = candidate.with_name("screening.json")
    candidate_manifest = _json(candidate)
    sources = candidate_manifest["sources"]
    if sources != _json(paths["screening"])["sources"]:
        raise ValueError("CFD screening and frozen candidates used different sources")
    if "split_audit_sha256" in candidate_manifest:
        paths["split_audit"] = candidate.with_name("split-audit.json")
    for kind, filename in (
        ("uiuc_index", "uiuc-index.html"),
        ("uiuc_coordinates", "uiuc-coordinates.zip"),
    ):
        source_path = candidate.with_name(filename)
        if _sha(source_path) != sources[kind]["sha256"]:
            raise ValueError(f"source snapshot changed: {source_path}")
        paths[kind] = source_path
    gpu = _checked_gpu_accounting(paths["gpu_budget"], spec["experiment"], runs)
    for directory, split, binding in (
        (official_dir, "test", {"test_stage_sha256": test_sha}),
        (transfer_dir, "matched_transfer", {"cohort_sha256": _sha(paths["cohorts"])}),
    ):
        metadata = _json(directory / "evaluation.json")
        if any(
            metadata.get(name) != value
            for name, value in (
                binding
                | {
                    "frozen_sha256": frozen_sha,
                    "selected_checkpoint": frozen["sha256"]["model.pt"],
                    "split": split,
                }
            ).items()
        ):
            raise ValueError(
                "evaluation differs from the frozen checkpoint or meshes at "
                f"{directory}"
            )

    official = summarise_official(
        _prediction_rows(official_dir, test_rows, transfer=False)
    )
    transfer, rankings = summarise_transfer(
        _prediction_rows(transfer_dir, cases, transfer=True)
    )
    _summary_matches(paths["official_summary"], official)
    _summary_matches(paths["transfer_summary"], transfer)
    _summary_matches(paths["rankings"], rankings)

    evidence = {
        "schema": 1,
        "experiment": spec["experiment"],
        "inputs": _source(spec_path),
        "runs": {
            role: {"run_id": record["run_id"], "record": record["record"]}
            for role, record in runs.items()
        },
        "artifacts": {
            **{
                role: _source(path) for role, path in paths.items() if not path.is_dir()
            },
            **{
                role: _source(paths[role] / "stage.json")
                for role in ("train_stage", "validation_stage", "test_stage")
            },
            "official_cases": _source(official_dir / "cases.jsonl"),
            "transfer_cases": _source(transfer_dir / "cases.jsonl"),
            "official_evaluation": _source(official_dir / "evaluation.json"),
            "transfer_evaluation": _source(transfer_dir / "evaluation.json"),
            "cfd_qualification": cohorts["qualification"],
            "training_metrics": _source(training / "metrics.jsonl"),
            "checkpoint_selection": _source(training / "selection.json"),
        },
        "cfd_reproductions": [
            {
                key: row[key]
                for key in ("case", "record", "verification", "cp_plot", "cp_data")
            }
            for row in reproduction
        ],
    }
    report = _render(
        out,
        evidence,
        official,
        transfer,
        rankings,
        {"parity": parity, "audit": audit, "reproduction": reproduction},
        {"gpu": gpu, "cfd": cohorts["budget"], "point": cases[0]},
        len(training_rows),
        len(validation_rows),
        run["config"]["seed"],
        frozen["selected"],
    )

    out.mkdir(parents=True, exist_ok=False)
    (out / "report.md").write_text(report)
    (out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )


def _number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4g}"


def _interval(bounds: list | None) -> str:
    return (
        "undefined"
        if bounds is None
        else f"[{_number(bounds[0])}, {_number(bounds[1])}]"
    )


def _table(header: tuple, rows: list[tuple]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def _error_rows(summary: dict, *, transfer: bool) -> list[tuple]:
    fields = (
        (
            ("volume_area_weighted_mse", "Area-weighted off-wall MSE"),
            ("surface_arc_length_weighted_mse", "Arc-length-weighted wall MSE"),
        )
        if transfer
        else (("volume_mse", "Off-wall MSE"), ("surface_mse", "Wall MSE"))
    )
    quantities = [(metric, label, summary["fields"]) for metric, label in fields]
    quantities.append(
        ("coefficient_absolute_error", "absolute error", summary["coefficients"])
    )
    if transfer:
        first, second = (summary["groups"][group] for group in ("naca", "uiuc"))
        delta = summary["uiuc_minus_naca"]["metrics"]
        estimate = "difference"
    else:
        first, second = (
            {"metrics": summary["metrics"][model]}
            for model in ("transolver", "baseline")
        )
        delta = summary["paired_transolver_minus_baseline"]
        estimate = "mean"
    rows = []
    for metric, label, channels in quantities:
        for i, channel in enumerate(channels):
            diff = delta[metric][i]
            supported = ""
            if not transfer:
                supported = "yes" if diff["supports_improvement"] else "uncertain"
                if diff["interval95"] is not None and diff["interval95"][0] > 0:
                    supported = "baseline lower"
            row = (
                f"{channel} {label}",
                _number(first["metrics"][metric]["mean"][i]),
                _number(second["metrics"][metric]["mean"][i]),
                f"{_number(diff[estimate])} {_interval(diff['interval95'])}",
            )
            rows.append(row if transfer else (*row, supported))
    for i, channel in enumerate(summary["coefficients"]):
        if transfer:
            diff = summary["uiuc_minus_naca"]["spearman"][i]
            interval = f"{_number(diff['difference'])} {_interval(diff['interval95'])}"
            rows.append(
                (
                    f"{channel} Spearman ρ",
                    _number(first["spearman"][i]),
                    _number(second["spearman"][i]),
                    interval,
                )
            )
        else:
            rows.append(
                (
                    f"{channel} Spearman ρ",
                    _number(summary["spearman"]["transolver"][i]),
                    _number(summary["spearman"]["baseline"][i]),
                    "",
                    "",
                )
            )
    return rows


def _render(
    out: Path,
    evidence: dict,
    official: dict,
    transfer: dict,
    rankings: dict,
    qualification: dict,
    allocations: dict,
    train_count: int,
    validation_count: int,
    seed: int,
    selected: dict,
) -> str:
    def link(source: dict, label: str) -> str:
        target = quote(os.path.relpath(source["path"], out))
        return f"[{label}]({target})"

    def artifact(role: str, label: str) -> str:
        return link(evidence["artifacts"][role], label)

    def registered(role: str) -> str:
        source = evidence["runs"][role]
        return link(source["record"], f"exp {source['run_id']}")

    audit = qualification["audit"]["cases"]
    adapter = np.abs([case["forces"]["adapter_minus_direct"] for case in audit]).max()
    stored = np.abs([case["forces"]["direct_minus_stored"] for case in audit]).max(
        axis=0
    )
    parity = qualification["parity"]
    parity_delta = max(
        max(row["max_abs_by_channel"]) for row in parity["checks"].values()
    )
    reproduction = [
        (
            " ".join(
                link(row[source], label)
                for source, label in (
                    ("verification", row["case"]),
                    ("cp_plot", "Cp profiles"),
                    ("record", "exp"),
                )
            ),
            *map(_number, row["coefficient_absolute_difference"] + row["cp_rmse"]),
        )
        for row in qualification["reproduction"]
    ]
    worst = []
    for group, metrics in rankings.items():
        for name, ranks in metrics.items():
            largest = max(
                ranks["ranks"],
                key=lambda row: abs(row["predicted_minus_reference_rank"]),
            )
            worst.append(
                (
                    group.upper(),
                    name,
                    largest["case"],
                    f"{largest['predicted_minus_reference_rank']:+g}",
                    len(ranks["discordant_pairs"]),
                )
            )

    point = allocations["point"]
    gpu_hours = allocations["gpu"]["charged_gpu_seconds"] / 3600
    cfd_hours = allocations["cfd"]["charged_core_seconds"] / 3600
    prep_hours = allocations["cfd"].get("preparation_core_seconds", 0) / 3600
    return "\n".join(
        [
            "# Original-mesh Transolver on AirfRANS",
            "",
            f"We trained one seed ({seed}) on {train_count} original AirfRANS cases. "
            f"We selected epoch {selected['epoch']} on {validation_count} "
            "validation cases "
            f"and scored the frozen checkpoint on {official['case_count']} test cases. "
            f"{registered('training')}, {registered('official')}, "
            f"{artifact('freeze', 'evaluation freeze')}.",
            "",
            "## Pipeline qualification",
            "",
            "The largest absolute FP32 forward difference was "
            f"{_number(parity_delta)}. {registered('qualification')}, "
            f"{artifact('parity', 'per-channel checks')}.",
            "",
            f"The {len(audit)} training meshes passed the wall and force checks. "
            f"The largest adapter discrepancy was {_number(adapter)}. The largest "
            f"stored-label differences were {_number(stored[0])} for CL and "
            f"{_number(stored[1])} for CD. "
            f"{registered('force_audit')}, {artifact('force_audit', 'force audit')}.",
            "",
            *_table(
                (
                    "CFD reproduction",
                    "Absolute CL difference",
                    "Absolute CD difference",
                    "Upper Cp RMSE",
                    "Lower Cp RMSE",
                ),
                reproduction,
            ),
            "",
            "## Official test",
            "",
            "Field baselines are training-set regional means. Force baselines are "
            "NACA-family regressions. Negative differences favour Transolver. "
            f"{registered('official')}, {registered('baselines')}, "
            f"{artifact('official_summary', 'full test metrics')}.",
            "",
            *_table(
                (
                    "Metric",
                    "Transolver",
                    "Baseline",
                    "Mean difference (95% interval)",
                    "Improvement supported",
                ),
                _error_rows(official, transfer=False),
            ),
            "",
            "## Transfer at matched conditions",
            "",
            f"We compared {transfer['groups']['uiuc']['case_count']} UIUC sections and "
            f"{transfer['groups']['naca']['case_count']} unseen NACA controls at "
            f"Re = {point['reynolds']:,.0f} and AoA = {point['aoa_deg']:+g}°. "
            "The intervals describe variation across these shapes for this checkpoint. "
            f"{registered('transfer')}, {artifact('cohorts', 'frozen CFD cases')}, "
            f"{artifact('transfer_summary', 'full transfer metrics')}.",
            "",
            *_table(
                ("Metric", "NACA", "UIUC", "UIUC minus NACA (95% interval)"),
                _error_rows(transfer, transfer=True),
            ),
            "",
            *_table(
                (
                    "Group",
                    "Coefficient",
                    "Largest rank displacement",
                    "Ranks shifted",
                    "Discordant pairs",
                ),
                worst,
            ),
            "",
            f"{artifact('rankings', 'All rank displacements, discordances and ties')}.",
            "",
            "## Reproduction",
            "",
            f"The campaign used {gpu_hours:.2f} A100-hours and {cfd_hours:.2f} "
            f"CFD CPU-core-hours, plus {prep_hours:.2f} CPU-core-hours "
            "for preparation. "
            f"{artifact('gpu_budget', 'A100 accounting')}, "
            f"{artifact('cohorts', 'CFD accounting')}.",
            "",
            f"{artifact('checkpoint', 'Selected checkpoint')}, "
            f"{artifact('normaliser', 'training normaliser')}, "
            f"{artifact('official_cases', 'official case rows')}, "
            f"{artifact('transfer_cases', 'transfer case rows')}, "
            f"{artifact('screening', 'geometry screening')}, "
            f"{artifact('protocol', 'protocol and commands')}, "
            "[evidence manifest](evidence.json).",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble_report(args.inputs, _json(PIN), PIN.parents[1], args.out)


if __name__ == "__main__":
    main()
