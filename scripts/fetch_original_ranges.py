"""Fetch the pinned ZIP in verified byte ranges on Eureka2 scratch."""

from __future__ import annotations

import argparse
import json
import re
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


def _available_ranges(previous: Path, size: int) -> list[tuple[int, int, Path]]:
    parts = []
    for path in previous.glob("range-*.part"):
        match = re.fullmatch(r"range-(\d+)-(\d+)\.part", path.name)
        if match is None:
            continue
        first, last = map(int, match.groups())
        headers = path.with_suffix(".headers").read_text().lower().splitlines()
        count = path.stat().st_size
        if (
            f"content-range: bytes {first}-{last}/{size}" in headers
            and 0 < count <= last - first + 1
            and 0 <= first <= last < size
        ):
            parts.append((first, first + count - 1, path))
    return parts


def download(
    prefix: Path,
    prefix_bytes: int,
    out: Path,
    workers: int,
    previous: tuple[Path, ...] = (),
) -> None:
    pin = json.loads(PIN.read_text())
    size = pin["size"]
    if not 0 < prefix_bytes < size or prefix.stat().st_size < prefix_bytes:
        raise ValueError("expected a complete prefix of the pinned ZIP")
    out.mkdir(parents=True, exist_ok=False)
    available = [(0, prefix_bytes - 1, prefix)]
    for source in previous:
        available.extend(_available_ranges(source, size))
    intervals, missing = [], []
    cursor = 0
    for first, last, path in sorted(available):
        if last < cursor:
            continue
        if first > cursor:
            missing.append((cursor, first - 1))
            cursor = first
        intervals.append((cursor, last, path, cursor - first))
        cursor = last + 1
    if cursor < size:
        missing.append((cursor, size - 1))

    chunk = (sum(last - first + 1 for first, last in missing) + workers - 1) // workers
    ranges = [
        (start, min(start + chunk - 1, last))
        for first, last in missing
        for start in range(first, last + 1, chunk)
    ]
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(fetch_range, pin, first, last, out) for first, last in ranges
        ]
        for (first, last), future in zip(ranges, futures, strict=True):
            intervals.append((first, last, future.result(), 0))

    partial = out / "Dataset.zip.assembling"
    with partial.open("xb") as target:
        for first, last, path, offset in sorted(intervals):
            with path.open("rb") as stream:
                stream.seek(offset)
                remaining = last - first + 1
                while remaining:
                    block = stream.read(min(remaining, BLOCK))
                    if not block:
                        raise ValueError(f"dataset byte range shortened: {path}")
                    target.write(block)
                    remaining -= len(block)
    if partial.stat().st_size != size:
        raise ValueError("assembled dataset has an unexpected size")
    partial.rename(out / "Dataset.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--prefix-bytes", type=int, required=True)
    parser.add_argument("--reuse-ranges", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    download(
        args.prefix,
        args.prefix_bytes,
        args.out,
        workers=4,
        previous=tuple(args.reuse_ranges),
    )


if __name__ == "__main__":
    main()
