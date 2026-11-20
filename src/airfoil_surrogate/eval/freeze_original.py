"""Freeze the selected checkpoint and training-only baselines for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import case_order_sha256
from airfoil_surrogate.train_original import OriginalTrainConfig


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze_evaluation(
    run_dir: Path,
    baselines: Path,
    pin: dict,
    out: Path,
    *,
    config: OriginalTrainConfig,
) -> None:
    """Check completed training and baseline provenance without accessing test data."""
    if out.exists():
        raise FileExistsError(out)
    artifacts = {
        name: (run_dir / name).read_bytes()
        for name in ("run.json", "selection.json", "normaliser.json", "metrics.jsonl")
    }
    artifacts["baselines"] = baselines.read_bytes()
    run = json.loads(artifacts["run.json"])
    if (
        run["config"] != json.loads(json.dumps(asdict(config)))
        or "timing_subset" in run["inputs"]
    ):
        raise ValueError("evaluation requires the complete prespecified training run")
    if set(run["inputs"].get("qualification", {})) != {
        "parity_sha256",
        "preflight_sha256",
        "force_audit_sha256",
    }:
        raise ValueError(
            "evaluation requires provenance from the training qualification"
        )
    selection = json.loads(artifacts["selection.json"])
    normaliser = json.loads(artifacts["normaliser.json"])
    baseline = json.loads(artifacts["baselines"])
    hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()
    }
    hashes["model.pt"] = _sha256(run_dir / "model.pt")
    if (
        run["normaliser_sha256"] != hashes["normaliser.json"]
        or selection["checkpoint_sha256"] != hashes["model.pt"]
    ):
        raise ValueError("training artefact checksum mismatch")

    expected = pin["partitions"]["train"]
    train_stage = run["inputs"]["train_stage_sha256"]
    fit_cases = normaliser["fit_cases"]
    if (
        normaliser["fit_stage_sha256"] != train_stage
        or baseline["fit_stage_sha256"] != train_stage
        or baseline["fit_cases"] != fit_cases
        or len(fit_cases) != expected["size"]
        or case_order_sha256(fit_cases) != expected["order_sha256"]
    ):
        raise ValueError(
            "the normaliser and baselines must match the pinned training stage"
        )

    epochs, interval = config.epochs, config.val_every
    scheduled = sorted(set(range(interval, epochs + 1, interval)) | {epochs})
    metrics = [json.loads(line) for line in artifacts["metrics.jsonl"].splitlines()]
    scores = [
        {"epoch": row["epoch"], "score": row["validation_score"]}
        for row in metrics
        if "validation_score" in row
    ]
    if (
        [row["epoch"] for row in metrics] != list(range(1, epochs + 1))
        or [score["epoch"] for score in scores] != scheduled
        or not all(math.isfinite(score["score"]) for score in scores)
        or scores != selection["validation_scores"]
    ):
        raise ValueError("training or validation history is incomplete or inconsistent")
    selected = min(scores, key=lambda score: (score["score"], score["epoch"]))
    if selected != {key: selection[key] for key in ("epoch", "score")}:
        raise ValueError(
            "selected checkpoint does not follow the recorded validation rule"
        )

    with out.open("x") as stream:
        json.dump(
            {
                "schema": 1,
                "training_stage_sha256": train_stage,
                "selected": selected,
                "sha256": hashes,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("baselines", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze_evaluation(
        args.training_run,
        args.baselines,
        json.loads(PIN.read_text()),
        args.out,
        config=OriginalTrainConfig(),
    )
    print(args.out)


if __name__ == "__main__":
    main()
