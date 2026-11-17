"""Prepare the three available original-mesh training probes for timing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from airfrans import Simulation

from airfoil_surrogate.data.features import NodeFeatures, Normaliser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    pin = json.loads((ROOT / "manifests/airfrans_remeshed.json").read_text())[
        "case_names"
    ]
    manifest = (ROOT / "data/raw/airfrans_remeshed" / pin["local_name"]).read_bytes()
    if hashlib.sha256(manifest).hexdigest() != pin["sha256"]:
        raise ValueError("original case manifest checksum mismatch")
    split = json.loads(manifest)
    train = split["full_train"][:720]
    cases = []
    sources = {}
    for index in (1, 2, 474):
        name = train[index]
        if name in split["full_test"]:
            raise ValueError("test case in pilot")
        sim = Simulation(str(args.dataset), name)
        x = np.column_stack((sim.position, sim.input_velocity, sim.sdf, -sim.normals))
        y = np.column_stack((sim.velocity, sim.pressure, sim.nu_t))
        if not np.isfinite(x).all() or not np.isfinite(y).all() or len(x) < 32000:
            raise ValueError(f"invalid pilot mesh: {name}")
        cases.append(NodeFeatures(sim.position, x, y, sim.surface))
        sources[name] = {
            suffix: hashlib.sha256(
                (args.dataset / name / f"{name}_{suffix}").read_bytes()
            ).hexdigest()
            for suffix in ("internal.vtu", "aerofoil.vtp")
        }
    norm = Normaliser.fit(cases)
    args.out.mkdir(parents=True, exist_ok=False)
    norm.save(args.out / "normaliser.json")
    for idx, case in enumerate(cases):
        np.savez(
            args.out / f"case{idx}.npz",
            x=norm.inputs(case.x).astype("float32"),
            pos=case.pos.astype("float32"),
            y=norm.targets(case.y).astype("float32"),
            surf=case.surf,
        )
    (args.out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
