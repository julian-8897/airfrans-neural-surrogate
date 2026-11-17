"""Compare stored prototype CL/CD with integration on both available meshes."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from airfrans import Simulation

from airfoil_surrogate.data.remeshed import (
    case_key,
    iter_samples,
    load_split_indices,
    match_case_names,
)
from airfoil_surrogate.eval.forces import RemeshedForceSimulation, force_coefficients
from airfoil_surrogate.train import ROOT, git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "original_root", type=Path, help="original AirfRANS Dataset root"
    )
    parser.add_argument("indices", type=int, nargs="+", help="indices from full_train")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/raw/airfrans_remeshed"
    )
    args = parser.parse_args()
    indices = set(args.indices)
    full_train = load_split_indices(args.data_dir / "README.md")["full_train"]
    if not indices <= set(full_train):
        parser.error("indices must belong to full_train")
    samples = list(iter_samples(args.data_dir, indices))
    if {sample.index for sample in samples} != indices:
        parser.error("some indices are missing from the dataset")
    names = match_case_names(
        (case_key(s.inlet_velocity, s.aoa) for s in samples),
        args.data_dir / "airfrans_manifest.json",
    )
    cases = []
    for sample, name in zip(samples, names, strict=True):
        original = Simulation(str(args.original_root), name)
        remeshed = RemeshedForceSimulation(sample)
        hashes = {
            patch: hashlib.sha256(
                (args.original_root / name / f"{name}_{patch}").read_bytes()
            ).hexdigest()
            for patch in ("internal.vtu", "aerofoil.vtp")
        }
        cases.append(
            {
                "index": sample.index,
                "case": name,
                "original_sha256": hashes,
                "stored": [sample.cl, sample.cd],
                "original": force_coefficients(original, reference=True).tolist(),
                "remeshed": force_coefficients(remeshed, reference=True).tolist(),
            }
        )
    result = {
        "task": "full",
        "split": "train",
        "dataset_dir": str(args.data_dir.resolve()),
        "coefficient_names": ["CL", "CD"],
        "contribution_names": ["total", "pressure", "viscous"],
        "git": git_state(),
        "versions": {p: version(p) for p in ("airfrans", "pyvista", "vtk")},
        "cases": cases,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
