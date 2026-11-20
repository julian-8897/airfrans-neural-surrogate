"""Estimate the original-mesh training cost using the production training path."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from airfoil_surrogate.data.features import Normaliser
from airfoil_surrogate.gpu_budget import TRAINING_MAX_SECONDS
from airfoil_surrogate.models.transolver_parity import verified_parity
from airfoil_surrogate.train_original import (
    OriginalTrainConfig,
    full_a100,
    pack_cases,
    train_original,
    training_inputs,
)

PROBE_EPOCHS = 11


def project_training(
    rows: list[dict],
    *,
    trial_cases: int,
    full_cases: int,
    training_load_seconds: float,
    validation_load_seconds: float,
    trial_seconds: float,
    config: OriginalTrainConfig,
) -> dict:
    training = [row["seconds"] for row in rows[1:] if "validation_score" not in row]
    validation = [row["seconds"] for row in rows if "validation_score" in row]
    rounds = (config.epochs + config.val_every - 1) // config.val_every
    scale = full_cases / trial_cases
    # The trial includes the exact epoch checkpoint, plus final checkpoint writes.
    seconds_outside_epoch = max(
        0.0, trial_seconds - sum(row["seconds"] for row in rows)
    )
    components = {
        "training_updates": max(training) * scale * config.epochs,
        "validation": max(0.0, max(validation) - statistics.median(training)) * rounds,
        "loading": training_load_seconds * scale + validation_load_seconds,
        "outside_epoch": seconds_outside_epoch / len(rows) * config.epochs,
    }
    total = sum(components.values())
    return {
        "components_seconds": components,
        "total_seconds": total,
        "max_training_seconds": TRAINING_MAX_SECONDS,
        "within_training_reservation": total <= TRAINING_MAX_SECONDS,
    }


def measure_preflight(
    train: Path,
    validation: Path,
    normaliser: Path,
    audit_path: Path,
    parity_path: Path,
    out: Path,
) -> bool:
    if out.exists():
        raise FileExistsError(out)
    train_rows, val_rows, norm_bytes, inputs = training_inputs(
        train, validation, normaliser
    )
    norm_sha = hashlib.sha256(norm_bytes).hexdigest()
    parity, parity_sha = verified_parity(
        parity_path, inputs["train_stage_sha256"], norm_sha
    )
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    audit_ids = [row["case"] for row in audit["cases"]]
    by_case = {row["id"]: row for row in train_rows}
    if (
        audit.get("schema") != 1
        or audit["passed"] is not True
        or audit["training_stage_sha256"] != inputs["train_stage_sha256"]
        or len(set(audit_ids)) != 12
        or len(audit_ids) != 12
        or not set(audit_ids) <= by_case.keys()
    ):
        raise ValueError(
            "preflight requires a passing 12-case audit on this training stage"
        )
    norm = Normaliser.from_dict(json.loads(norm_bytes))
    device = full_a100()
    gpu = torch.cuda.get_device_name(device)
    if (gpu, str(torch.__version__), torch.version.cuda) != (
        parity["gpu"],
        parity["versions"]["torch"],
        parity["cuda"],
    ):
        raise ValueError("parity and timing need the same A100 model and PyTorch build")
    torch.set_float32_matmul_precision("highest")
    start = time.perf_counter()
    training = pack_cases(train, [by_case[case] for case in audit_ids], norm)
    training_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    validating = pack_cases(validation, val_rows, norm)
    validation_load_seconds = time.perf_counter() - start

    out.mkdir(parents=True, exist_ok=False)
    config = OriginalTrainConfig()
    trial = out / "timing-attempt"
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    train_original(
        replace(config, epochs=PROBE_EPOCHS),
        training,
        validating,
        device=device,
        out=trial,
        normaliser_bytes=norm_bytes,
        inputs=inputs | {"timing_subset": audit_ids},
    )
    torch.cuda.synchronize(device)
    trial_seconds = time.perf_counter() - start
    rows = [
        json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()
    ]
    estimate = project_training(
        rows,
        trial_cases=len(training),
        full_cases=len(train_rows),
        training_load_seconds=training_load_seconds,
        validation_load_seconds=validation_load_seconds,
        trial_seconds=trial_seconds,
        config=config,
    )
    with (out / "preflight.json").open("x") as stream:
        json.dump(
            {
                "schema": 1,
                "inputs": inputs,
                "normaliser_sha256": norm_sha,
                "parity_sha256": parity_sha,
                "force_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "production_config": asdict(config),
                "environment": {
                    "gpu": gpu,
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                },
                "timing_case_ids": audit_ids,
                "validation_cases": len(val_rows),
                "trial_seconds": trial_seconds,
                "trial_training_load_seconds": training_load_seconds,
                "trial_validation_load_seconds": validation_load_seconds,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "projection": estimate,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    return estimate["within_training_reservation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("normaliser", type=Path)
    parser.add_argument("force_audit", type=Path)
    parser.add_argument("parity", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not measure_preflight(
        args.train,
        args.validation,
        args.normaliser,
        args.force_audit,
        args.parity,
        args.out,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
