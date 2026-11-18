"""Stage one official AirfRANS mesh partition from the pinned authors' ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

PIN = Path(__file__).resolve().parents[3] / "manifests/airfrans_original.json"
PARTITIONS = ("train", "validation", "test")
MESH_SUFFIXES = ("internal.vtu", "aerofoil.vtp")


def zip_index_sha256(archive: ZipFile) -> str:
    """Hash the sorted ZIP catalogue of (member name, size, CRC-32) triples."""
    entries = sorted((m.filename, m.file_size, m.CRC) for m in archive.infolist())
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode()
    ).hexdigest()


def _stage_file(archive: ZipFile, member: ZipInfo, destination: Path) -> str:
    digest = hashlib.sha256()
    # ZipExtFile verifies CRC-32 when the member is read to its end.
    with archive.open(member) as source, destination.open("xb") as out:
        while chunk := source.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def stage_partition(
    archive: ZipFile,
    pin: dict,
    partition: str,
    out: Path,
    *,
    case_ids: list[str] | None = None,
) -> Path:
    """Extract only the chosen mesh members into a create-only staging directory."""
    if zip_index_sha256(archive) != pin["zip_index_sha256"]:
        raise ValueError("original AirfRANS ZIP catalogue checksum mismatch")
    raw_splits = archive.read(pin["split_path"])
    if hashlib.sha256(raw_splits).hexdigest() != pin["split_sha256"]:
        raise ValueError("original AirfRANS split checksum mismatch")
    official = json.loads(raw_splits)
    full_train, full_test = official["full_train"], official["full_test"]
    all_ids = full_train + full_test
    if (
        len(full_train) != 800
        or len(full_test) != 200
        or not all(
            isinstance(case_id, str)
            and case_id.startswith("airFoil2D_SST_")
            and "/" not in case_id
            and "\\" not in case_id
            for case_id in all_ids
        )
        or len(set(all_ids)) != 1000
    ):
        raise ValueError(
            "official full-task split must contain 1,000 distinct case IDs"
        )
    partitions = {
        "train": full_train[:720],
        "validation": full_train[720:],
        "test": full_test,
    }
    ordered = partitions[partition]
    chosen = ordered
    if case_ids is not None:
        requested = set(case_ids)
        if not requested or len(requested) != len(case_ids) or requested - set(ordered):
            raise ValueError(
                f"expected distinct case IDs from the {partition} partition"
            )
        chosen = [case_id for case_id in ordered if case_id in requested]

    # Resolve the selected members before making the output directory.
    prefix = PurePosixPath(pin["split_path"]).parent
    members = {
        case_id: {
            suffix: archive.getinfo(str(prefix / case_id / f"{case_id}_{suffix}"))
            for suffix in MESH_SUFFIXES
        }
        for case_id in chosen
    }
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for case_id, mesh_files in members.items():
        case_dir = out / case_id
        case_dir.mkdir()
        files = {}
        for suffix, member in mesh_files.items():
            file_path = case_dir / f"{case_id}_{suffix}"
            files[suffix] = {
                "size": member.file_size,
                "sha256": _stage_file(archive, member, file_path),
            }
        cases.append({"id": case_id, "files": files})

    manifest = out / "stage.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": pin,
                "partition": partition,
                "partition_size": len(ordered),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="local authors' Dataset.zip")
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory")
    parser.add_argument(
        "--case", action="append", help="optional case IDs within partition"
    )
    args = parser.parse_args()

    pin = json.loads(PIN.read_text())
    if args.archive.stat().st_size != pin["size"]:
        parser.error("original AirfRANS ZIP size mismatch")
    with ZipFile(args.archive) as archive:
        print(
            stage_partition(archive, pin, args.partition, args.out, case_ids=args.case)
        )


if __name__ == "__main__":
    main()
