"""Verify and report the retained matched-cohort evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.eval.original import frozen_inputs
from airfoil_surrogate.eval.study_report import (
    _cfd_reports,
    _error_rows,
    _json,
    _prediction_rows,
    _sha,
    _source,
    _summary_matches,
    _table,
)
from airfoil_surrogate.eval.transfer import frozen_cases
from airfoil_surrogate.eval.transfer_metrics import summarise_transfer


def assemble(spec_path: Path, root: Path, out: Path) -> None:
    spec = _json(spec_path)
    study = spec["experiment"]
    experiment_path = root / "experiments" / study / "experiment.json"
    experiment = _json(experiment_path)
    run_path = experiment_path.parent / "runs" / spec["transfer_run_id"] / "run.json"
    run = _json(run_path)
    if (
        experiment["id"] != study
        or run["experiment"] != study
        or run["run_id"] != spec["transfer_run_id"]
        or run["status"] != "succeeded"
        or run["backend"]["kind"] != "slurm"
    ):
        raise ValueError("expected the successful registered matched inference")

    training, baselines, freeze_path, cohorts_path = (
        Path(spec[key]) for key in ("training", "baselines", "freeze", "cohorts")
    )
    frozen = _json(freeze_path)
    frozen_inputs(training, baselines, frozen)
    cohorts = _json(cohorts_path)
    cases = frozen_cases(cohorts, _json(PIN))
    references = _cfd_reports(
        cohorts,
        cases,
        study,
        frozen["training_stage_sha256"],
        reference_experiment=experiment["parent"],
    )
    evaluation = root / run["output_dir"] / "transfer"
    metadata_path = evaluation / "evaluation.json"
    metadata = _json(metadata_path)
    for key, value in {
        "split": "matched_transfer",
        "cases": [[case["id"], case["group"]] for case in cases],
        "cohort_sha256": _sha(cohorts_path),
        "frozen_sha256": _sha(freeze_path),
        "selected_checkpoint": frozen["sha256"]["model.pt"],
        "operating_point": {
            key: cases[0][key] for key in ("inlet_velocity", "aoa_deg", "reynolds")
        },
    }.items():
        if metadata.get(key) != value:
            raise ValueError(f"matched evaluation provenance differs at {key}")
    summary, rankings = summarise_transfer(
        _prediction_rows(evaluation, cases, transfer=True)
    )
    summary_path = evaluation / "report/summary.json"
    rankings_path = evaluation / "report/rankings.json"
    cases_path = evaluation / "cases.jsonl"
    _summary_matches(summary_path, summary)
    _summary_matches(rankings_path, rankings)

    def link(path: Path, text: str) -> str:
        relative = quote(os.path.relpath(path.resolve(), out.resolve()))
        return f"[{text}]({relative})"

    point = metadata["operating_point"]
    allocation = cohorts["budget"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.md").write_text(
        "\n".join(
            [
                "# Frozen Transolver on matched airfoil geometries",
                "",
                f"We evaluated the frozen epoch {frozen['selected']['epoch']} "
                f"checkpoint on 30 UIUC and 30 unseen NACA airfoils at "
                f"Re = {point['reynolds']:.3g} and AoA = {point['aoa_deg']:g}°. "
                f"{link(run_path, 'Registered inference')}.",
                "",
                "Differences are UIUC minus NACA. Field MSE uses the frozen "
                "training scales. Intervals use independent shape bootstraps "
                "within each group. The comparison is conditional on CFD "
                "qualification and one trained checkpoint.",
                "",
                *_table(
                    ("Metric", "NACA", "UIUC", "Difference (95% interval)"),
                    _error_rows(summary, transfer=True),
                ),
                "",
                f"{link(summary_path, 'Full metrics')} · "
                f"{link(rankings_path, 'Rank errors and discordant pairs')} · "
                f"{link(cases_path, 'Case rows and prediction hashes')} · "
                f"{link(cohorts_path, 'Frozen meshes and allocation evidence')}",
                "",
            ]
        )
    )
    artefacts = (
        run_path,
        training / "model.pt",
        training / "normaliser.json",
        baselines,
        freeze_path,
        cohorts_path,
        metadata_path,
        cases_path,
        summary_path,
        rankings_path,
    )
    (out / "evidence.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "experiment": study,
                "case_count": len(cases),
                "inputs": _source(spec_path),
                "artifacts": [_source(path) for path in artefacts],
                "reproduction_reports": [row["verification"] for row in references],
                "cpu_core_seconds_through_matched_cfd": (
                    allocation["charged_core_seconds"]
                    + allocation.get("preparation_core_seconds", 0)
                ),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.inputs, Path.cwd(), args.out)


if __name__ == "__main__":
    main()
