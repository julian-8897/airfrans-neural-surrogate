"""Report the official test and the documented matched-CFD gate failure."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

from airfoil_surrogate.cfd.accounting import (
    TERMINAL,
    TOTAL_CORE_SECONDS,
    read_allocations,
)
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import read_complete_partition
from airfoil_surrogate.eval.original import frozen_inputs, summarise_official
from airfoil_surrogate.eval.study_report import (
    _error_rows,
    _json,
    _number,
    _prediction_rows,
    _sha,
    _source,
    _summary_matches,
    _table,
)
from airfoil_surrogate.gpu_budget import GPU_CAP_SECONDS, read_gpu_allocations

ROLES = (
    "training",
    "baselines",
    "official_stage",
    "official",
    "force_audit",
    "qualification",
)


def _sacct(ids: list[str], fields: str, out: Path) -> str:
    if not ids or len(set(ids)) != len(ids) or not all(job.isdecimal() for job in ids):
        raise ValueError("expected distinct allocation IDs for the study")
    rows = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "--duplicates", "-j", ",".join(ids), "-o", fields],
        text=True,
    )
    if set(ids) != {row.partition("|")[0].strip() for row in rows.splitlines()}:
        raise ValueError("accounting does not cover the registered allocations")
    with out.open("x") as stream:
        stream.write(rows)
    return rows


def assemble(spec_path: Path, root: Path, out: Path) -> None:
    spec = _json(spec_path)
    study, runs = spec["experiment"], spec["runs"]
    if set(runs) != {*ROLES, "reproduction_gate"}:
        raise ValueError("missing required run provenance")
    output, records = {}, {}
    for role, link in runs.items():
        run_id = link["run_id"]
        record = root / "experiments" / study / "runs" / run_id / "run.json"
        metadata = _json(record)
        if metadata["experiment"] != study or metadata["run_id"] != run_id:
            raise ValueError("run record belongs to a different study")
        if role in ROLES and (
            metadata["backend"]["kind"] != "slurm"
            or metadata["backend"].get("job_id") not in (None, link["job_id"])
        ):
            raise ValueError(f"allocation does not match the registered {role} run")
        output[role] = root / metadata["output_dir"]
        records[role] = _source(record)

    training = output["training"] / "training"
    staged = output["official_stage"]
    test_dir = output["official"] / "test"
    freeze_path = staged / "frozen.json"
    frozen = _json(freeze_path)
    norm = training / "normaliser.json"
    checkpoint = training / "model.pt"
    baselines = output["baselines"] / "baselines.json"
    frozen_inputs(training, baselines, frozen)
    rows, stage_sha = read_complete_partition(
        staged / "test", "test", _json(PIN), frozen_sha256=_sha(freeze_path)
    )
    evaluation_path = test_dir / "evaluation.json"
    metadata = _json(evaluation_path)
    if (
        metadata["test_stage_sha256"] != stage_sha
        or metadata["frozen_sha256"] != _sha(freeze_path)
        or metadata["selected_checkpoint"] != _sha(checkpoint)
        or metadata["split"] != "test"
    ):
        raise ValueError(
            "test evaluation differs from the frozen production checkpoint"
        )

    official = summarise_official(_prediction_rows(test_dir, rows, transfer=False))
    summary_path = test_dir / "summary.json"
    _summary_matches(summary_path, official)
    gate_path = output["reproduction_gate"] / "reproduction-gate.md"
    gate = _json(output["reproduction_gate"] / "metrics.json")
    if gate["reproduction_gate_passed"] is not False:
        raise ValueError("expected a documented failure of the matched CFD gate")

    out.mkdir(exist_ok=False, parents=True)
    cpu_path, gpu_path = out / "cpu-accounting.psv", out / "gpu-accounting.psv"
    cpu = read_allocations(
        _sacct(
            spec["cpu_jobs"],
            "JobIDRaw,AllocCPUS,ReqCPUS,ElapsedRaw,TimelimitRaw,State%40,ExitCode",
            cpu_path,
        ),
        required_cpus=None,
    )
    gpu = read_gpu_allocations(
        _sacct(
            spec["gpu_jobs"],
            "JobIDRaw,AllocTRES%256,ReqTRES%256,ElapsedRaw,TimelimitRaw,State%40,ExitCode",
            gpu_path,
        )
    )
    for role in ROLES:
        allocation = cpu[runs[role]["job_id"]]
        if (allocation.state, allocation.exit_code) != ("COMPLETED", "0:0"):
            raise ValueError(f"missing successful Slurm allocation for {role}")
    if not all(row.state in TERMINAL for row in cpu.values()):
        raise ValueError("expected terminal study allocations")
    if not all(
        row.succeeded
        for row in (
            gpu[runs[key]["job_id"]]
            for key in ("qualification", "training", "official")
        )
    ):
        raise ValueError(
            "expected successful qualification, training and inference GPUs"
        )
    cpu_seconds = sum(row.charge for row in cpu.values())
    gpu_seconds = sum(row.charge for row in gpu.values())
    if (
        cpu_seconds + spec["report_reserved_core_seconds"] > TOTAL_CORE_SECONDS
        or gpu_seconds > GPU_CAP_SECONDS
    ):
        raise ValueError("study allocations exceeded the approved budget")

    def link(path: Path, label: str) -> str:
        return f"[{label}]({quote(os.path.relpath(path.resolve(), out.resolve()))})"

    def registered(role: str) -> str:
        source = records[role]
        return link(Path(source["path"]), f"exp {runs[role]['run_id']}")

    selected = frozen["selected"]
    gate_detail = (
        f"{gate['mesh_failures']} mesh checks failed and "
        f"{gate['solver_timeouts']} solver reached the allocation limit."
    )
    lines = [
        "# Original-mesh Transolver on AirfRANS",
        "",
        f"Seed 0 selected epoch {selected['epoch']} by its validation score "
        f"({_number(selected['score'])}). We evaluated the frozen checkpoint on "
        f"{official['case_count']} official meshes. "
        f"{registered('training')}, {registered('official')}.",
        "",
        "## Pipeline qualification",
        "",
        "The original force audit and pinned same-weight A100 forward comparison "
        f"passed. {registered('force_audit')}, {registered('qualification')}.",
        "",
        "## Official test",
        "",
        "The field baseline uses training regional means. The force baseline is "
        "the NACA-family regression. Negative differences favour Transolver. "
        f"{registered('baselines')}, {link(summary_path, 'full test summary')}.",
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
        "## Matched CFD",
        "",
        "The approved reproduction gate failed. " + gate_detail + " "
        f"{link(gate_path, 'Reproduction evidence')}, "
        f"{registered('reproduction_gate')}.",
        "",
        "## Artefacts",
        "",
        f"{link(checkpoint, 'Selected checkpoint')}, {link(norm, 'normaliser')}, "
        f"{link(freeze_path, 'evaluation freeze')}, "
        f"{link(staged / 'test/stage.json', 'test manifest')}, "
        f"{link(test_dir / 'cases.jsonl', 'case metrics and prediction hashes')}.",
        "",
        "Through official inference, Slurm reports "
        f"{gpu_seconds / 3600:.2f} A100-hours and {cpu_seconds / 3600:.2f} "
        "allocated CPU-core-hours including GPU host CPUs. "
        f"{link(gpu_path, 'GPU rows')}, {link(cpu_path, 'CPU rows')}.",
        "",
    ]
    files = (
        checkpoint,
        norm,
        baselines,
        freeze_path,
        staged / "test/stage.json",
        evaluation_path,
        test_dir / "cases.jsonl",
        summary_path,
        gate_path,
        cpu_path,
        gpu_path,
    )
    (out / "report.md").write_text("\n".join(lines))
    (out / "evidence.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "experiment": study,
                "runs": {role: record | runs[role] for role, record in records.items()},
                "inputs": _source(spec_path),
                "artefacts": [_source(path) for path in files],
                "matched_reproduction_gate_passed": False,
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
    assemble(args.inputs, PIN.parents[1], args.out)


if __name__ == "__main__":
    main()
