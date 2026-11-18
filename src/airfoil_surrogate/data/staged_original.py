"""Verify staged original AirfRANS data before use by training or evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from airfoil_surrogate.data.stage_original import MESH_SUFFIXES


def case_order_sha256(case_ids: list[str]) -> str:
    """Hash the case IDs as an ordered, compact JSON list."""
    return hashlib.sha256(
        json.dumps(case_ids, separators=(",", ":")).encode()
    ).hexdigest()


def read_complete_partition(
    root: Path, partition: str, pin: dict, *, frozen_sha256: str | None = None
) -> tuple[list[dict], str]:
    """Check membership and order without opening any mesh files."""
    manifest = (root / "stage.json").read_bytes()
    metadata = json.loads(manifest)
    if metadata["schema"] != 1 or metadata["partition"] != partition:
        raise ValueError(f"expected a staged {partition} partition")
    if partition == "test" and (
        frozen_sha256 is None or metadata.get("frozen_sha256") != frozen_sha256
    ):
        raise ValueError("test stage was not made for this evaluation freeze")
    if any(
        metadata["source"][digest] != pin[digest]
        for digest in ("zip_index_sha256", "split_sha256")
    ):
        raise ValueError("staged meshes do not identify the pinned original dataset")

    rows = metadata["cases"]
    expected = pin["partitions"][partition]
    if (
        metadata["partition_size"] != expected["size"]
        or len(rows) != expected["size"]
        or case_order_sha256([row["id"] for row in rows]) != expected["order_sha256"]
    ):
        raise ValueError(f"staged cases must match the full official {partition} order")
    return rows, hashlib.sha256(manifest).hexdigest()


def verified_mesh_paths(root: Path, row: dict) -> tuple[Path, Path]:
    """Verify both files of a case from a checked partition manifest."""
    case_id = row["id"]
    paths = []
    for suffix in MESH_SUFFIXES:
        expected = row["files"][suffix]
        path = root / case_id / f"{case_id}_{suffix}"
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"size mismatch for staged mesh {path}")
        with path.open("rb") as data:
            actual = hashlib.file_digest(data, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"checksum mismatch for staged mesh {path}")
        paths.append(path)
    return paths[0], paths[1]
