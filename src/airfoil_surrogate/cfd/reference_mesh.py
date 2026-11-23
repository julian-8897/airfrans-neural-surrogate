"""Apply the approved aspect-ratio exception to two pinned reference meshes."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

PIN = Path(__file__).resolve().parents[3] / "manifests/airfrans_reproduction_mesh.json"


def approved_reference_mesh(case_id: str, plan_sha256: str) -> dict:
    raw = PIN.read_bytes()
    pin = json.loads(raw)
    if plan_sha256 != pin["plan_sha256"] or case_id not in pin["cases"]:
        raise ValueError("reference mesh exception is outside the approved cases")
    return pin["cases"][case_id] | {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_reference_mesh(log: str, mesh: Path, evidence: dict) -> None:
    matches = re.findall(
        r"^\s*\*\*\*High aspect ratio cells found, Max aspect ratio: ([\d.]+), "
        r"number of cells (\d+)\s*$",
        log,
        re.M,
    )
    if (
        matches != [(evidence["max_aspect_ratio"], str(evidence["flagged_cells"]))]
        or re.findall(r"^\s*Failed (\d+) mesh checks\.\s*$", log, re.M) != ["1"]
        or "FOAM FATAL" in log
    ):
        raise ValueError("expected only the pinned high aspect ratio check failure")
    for name, expected in evidence["poly_mesh_sha256"].items():
        path = mesh / name
        open_mesh = gzip.open if path.suffix == ".gz" else open
        with open_mesh(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"reference mesh differs from the audited mesh: {name}")
