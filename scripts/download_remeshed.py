"""Download AirfRANS_remeshed and the AirfRANS case-name manifest, then verify.

Revisions and SHA-256 checksums are pinned in manifests/airfrans_remeshed.json.

    uv run python scripts/download_remeshed.py [--out data/raw/airfrans_remeshed]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "airfrans_remeshed.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, size: int, digest: str) -> None:
    if path.stat().st_size != size or sha256(path) != digest:
        raise RuntimeError(f"checksum mismatch for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/airfrans_remeshed")
    args = parser.parse_args()
    spec = json.loads(MANIFEST.read_text())

    snapshot_download(
        spec["dataset"],
        repo_type="dataset",
        revision=spec["revision"],
        local_dir=args.out,
        allow_patterns=list(spec["files"]),
    )
    for rel, meta in spec["files"].items():
        verify(args.out / rel, meta["size"], meta["sha256"])

    names = spec["case_names"]
    target = args.out / names["local_name"]
    if not target.exists():
        cached = hf_hub_download(
            names["dataset"],
            names["path"],
            repo_type="dataset",
            revision=names["revision"],
        )
        shutil.copyfile(cached, target)
    verify(target, names["size"], names["sha256"])
    print(f"verified {len(spec['files']) + 1} files in {args.out}")


if __name__ == "__main__":
    main()
