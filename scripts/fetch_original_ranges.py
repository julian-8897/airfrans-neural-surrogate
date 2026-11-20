"""Fetch the pinned ZIP in verified byte ranges on Eureka2 scratch."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PIN = Path(__file__).resolve().parents[1] / "manifests/airfrans_original.json"
BLOCK = 1 << 20


def fetch_range(pin: dict, first: int, last: int, out: Path) -> Path:
    path = out / f"range-{first}-{last}.part"
    headers = path.with_suffix(".headers")
    result = subprocess.check_output(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            "--connect-timeout",
            "20",
            "--header",
            f"If-Match: {pin['etag']}",
            "--range",
            f"{first}-{last}",
            "--dump-header",
            str(headers),
            "--write-out",
            "%{http_code}",
            "--output",
            str(path),
            pin["url"],
        ],
        text=True,
    )
    expected = f"content-range: bytes {first}-{last}/{pin['size']}"
    if (
        result != "206"
        or expected not in headers.read_text().lower().splitlines()
        or path.stat().st_size != last - first + 1
    ):
        raise ValueError(f"invalid dataset response for bytes {first}–{last}")
    print(f"Verified HTTP range {first}–{last}", flush=True)
    return path


def download(prefix: Path, prefix_bytes: int, out: Path, workers: int) -> None:
    pin = json.loads(PIN.read_text())
    size = pin["size"]
    if not 0 < prefix_bytes < size or prefix.stat().st_size < prefix_bytes:
        raise ValueError("expected a complete prefix of the pinned ZIP")
    out.mkdir(parents=True, exist_ok=False)
    chunk = (size - prefix_bytes + workers - 1) // workers
    ranges = [
        (start, min(start + chunk, size) - 1)
        for start in range(prefix_bytes, size, chunk)
    ]
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(fetch_range, pin, first, last, out) for first, last in ranges
        ]
        parts = [future.result() for future in futures]

    partial = out / "Dataset.zip.assembling"
    with partial.open("xb") as target:
        with prefix.open("rb") as stream:
            remaining = prefix_bytes
            while remaining:
                block = stream.read(min(remaining, BLOCK))
                if not block:
                    raise ValueError("dataset prefix shortened while being copied")
                target.write(block)
                remaining -= len(block)
        for path in parts:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, target, BLOCK)
    if partial.stat().st_size != size:
        raise ValueError("assembled dataset has an unexpected size")
    partial.rename(out / "Dataset.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--prefix-bytes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    download(args.prefix, args.prefix_bytes, args.out, workers=4)


if __name__ == "__main__":
    main()
