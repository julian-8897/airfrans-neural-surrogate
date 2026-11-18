"""Evaluate an AirfRANS_remeshed checkpoint on its saved validation cases."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from airfoil_surrogate.data.features import TARGET_NAMES, Normaliser, build_features
from airfoil_surrogate.data.remeshed import iter_samples, load_split_indices
from airfoil_surrogate.eval.metrics import field_errors, json_values, summarise
from airfoil_surrogate.models.transolver import Transolver, TransolverConfig
from airfoil_surrogate.train import ROOT, git_state, pick_device


@torch.inference_mode()
def evaluate_run(
    run_dir: Path,
    out: Path,
    *,
    device_name: str = "cpu",
    limit: int | None = None,
    forces: bool = False,
) -> Path:
    if forces:
        try:
            from airfoil_surrogate.eval.forces import coefficient_spearman, force_errors
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Run with `uv run --extra forces` to enable forces"
            ) from exc

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    run = json.loads((run_dir / "run.json").read_text())
    config = run["config"]
    task = config["task"]
    data_dir = ROOT / config["data_dir"]
    splits = load_split_indices(data_dir / "README.md")
    val_ids = run["val_ids"]
    if (
        not val_ids
        or len(val_ids) != len(set(val_ids))
        or not set(val_ids) <= set(splits[f"{task}_train"])
        or set(val_ids) & (set(run["train_ids"]) | set(splits[f"{task}_test"]))
    ):
        raise ValueError("checkpoint metadata has invalid validation indices")
    val_ids = val_ids[:limit]

    normaliser = Normaliser.load(run_dir / "normaliser.json")
    device = pick_device(device_name)
    checkpoint = torch.load(run_dir / "model.pt", weights_only=True, map_location="cpu")
    model = Transolver(TransolverConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    by_case = {}
    for sample in iter_samples(data_dir, val_ids):
        features = build_features(sample)
        x = torch.as_tensor(
            normaliser.inputs(features.x), dtype=torch.float32, device=device
        )[None]
        pos = torch.as_tensor(features.pos, dtype=torch.float32, device=device)[None]
        prediction = model(x, pos)[0].cpu().numpy().astype(np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"sample {sample.index}: non-finite predictions")
        metrics = field_errors(prediction, features.y, features.surf, normaliser)
        if forces:
            metrics.update(force_errors(sample, normaliser.inverse_targets(prediction)))
        by_case[sample.index] = metrics
    if set(by_case) != set(val_ids):
        raise ValueError(
            f"missing validation cases: {sorted(set(val_ids) - by_case.keys())}"
        )

    rows = [by_case[i] for i in val_ids]
    columns = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    summary = {
        "dataset": "AirfRANS_remeshed",
        "run_dir": str(run_dir.resolve()),
        "dataset_dir": str(data_dir.resolve()),
        "task": task,
        "split": "validation",
        "case_count": len(rows),
        "target_names": TARGET_NAMES,
        "device": str(device),
        "torch": str(torch.__version__),
        "git": git_state(),
        "metrics": {
            key: summarise(values)
            for key, values in columns.items()
            if key.endswith(("mse", "relative_l2", "relative_error"))
        },
    }
    if forces:
        predicted = columns["coefficients_predicted"][:, :, 0]
        reference = columns["coefficients_reference"][:, :, 0]
        stored = columns["coefficients_stored"]
        summary["force_diagnostics"] = {
            "versions": {p: version(p) for p in ("airfrans", "pyvista", "vtk")},
            "coefficient_names": ["CL", "CD"],
            "contribution_names": ["total", "pressure", "viscous"],
            "spearman_vs_remeshed_reference": coefficient_spearman(
                predicted, reference
            ),
            "spearman_vs_stored": coefficient_spearman(predicted, stored),
            "reference_absolute_difference_vs_stored": summarise(
                np.abs(columns["coefficients_reference_minus_stored"])
            ),
            "wss_components": ["x", "y"],
            "wss_rmse": summarise(columns["wss_rmse"]),
        }

    out.mkdir(parents=True, exist_ok=False)
    with (out / "cases.jsonl").open("x") as handle:
        for index, row in zip(val_ids, rows, strict=True):
            record = {"index": index, **{k: json_values(v) for k, v in row.items()}}
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="new directory for evaluation outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit", type=int, help="evaluate the first N validation cases"
    )
    parser.add_argument(
        "--forces", action="store_true", help="add prototype force diagnostics"
    )
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "runs" / f"{stamp}-eval-{args.run_dir.resolve().name}"
    print(
        evaluate_run(
            args.run_dir,
            out,
            device_name=args.device,
            limit=args.limit,
            forces=args.forces,
        )
    )


if __name__ == "__main__":
    main()
