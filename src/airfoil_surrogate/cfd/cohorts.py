"""Screen and rank the matched UIUC and unseen NACA geometry cohorts."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from airfrans.naca_generator import naca_generator
from scipy.spatial import cKDTree

from airfoil_surrogate.cfd.geometry import (
    Profile,
    align_profile,
    parse_coordinates,
    passes_shape_screen,
    split_contour,
)
from airfoil_surrogate.cfd.reproduction import GENERATOR_REVISION
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.eval.forces import AIRFRANS_VISCOSITY

UIUC_INDEX = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
UIUC_ARCHIVE = "https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip"
CASE_ORDER_KEY = "airfrans-transfer-v1/"
NEAR_DUPLICATE = 0.001


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def selection_key(label: str) -> str:
    return _sha256((CASE_ORDER_KEY + label).encode())


def _index_labels(index: bytes) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for filename, description in re.findall(
        r'<a\b[^>]*\bhref=["\'](?:coord|coord_updates)/([^/"\']+\.dat)["\']'
        r"[^>]*>[^<]*</a>\s*\\\s*([^\\<\r\n]+)",
        index.decode("latin-1"),
        flags=re.IGNORECASE,
    ):
        labels[filename.lower()].add(html.unescape(description).strip())
    return labels


def _rms_vector(profile: Profile) -> np.ndarray:
    values = profile.ordinate_samples.ravel()
    return values / np.sqrt(len(values))


def _shape_record(profile: Profile) -> dict:
    return {
        "thickness": profile.thickness,
        "absolute_camber": profile.absolute_camber,
        "trailing_gap": profile.trailing_gap,
    }


def _identity_issue(filename: str, label: str, index_labels: list[str]) -> str | None:
    words = " ".join((filename, label, *index_labels)).lower()
    if "naca" in words:
        return "named NACA or NACA-derived profile"
    if re.search(
        r"multi[ -]?element|\bslat\b|\bmain element\b|\bflap element\b", words
    ):
        return "named multi-element profile or component"
    if (
        not label
        or not index_labels
        or re.search(r"\b(unknown|unnamed|unidentified)\b", words)
    ):
        return "missing or ambiguous identity"
    return None


def screen_uiuc(
    index: bytes, archive: ZipFile, naca: dict[str, Profile]
) -> tuple[list[dict], list[tuple[dict, Profile]]]:
    """Audit every source file and greedily retain unique eligible UIUC profiles."""
    labels = _index_labels(index)
    tree = cKDTree(np.stack([_rms_vector(profile) for profile in naca.values()]))
    naca_ids = tuple(naca)
    coordinate_members = [
        info
        for info in archive.infolist()
        if not info.is_dir() and info.filename.lower().endswith(".dat")
    ]
    entries = {
        PurePosixPath(info.filename).name.lower(): info for info in coordinate_members
    }
    if len(entries) != len(coordinate_members):
        raise ValueError("UIUC archive contains duplicate case-insensitive filenames")
    audited, retained, vectors = [], [], []
    for filename in sorted(entries, key=selection_key):
        raw = archive.read(entries[filename])
        row = {
            "filename": filename,
            "source_sha256": _sha256(raw),
            "index_labels": sorted(labels.get(filename, ())),
            "order_sha256": selection_key(filename),
        }
        audited.append(row)
        try:
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.dat", filename):
                raise ValueError("ambiguous source filename")
            label, first, second = parse_coordinates(raw)
            row["coordinate_label"] = label
            issue = _identity_issue(filename, label, row["index_labels"])
            if issue:
                raise ValueError(issue)
            profile = align_profile(first, second)
            row.update(_shape_record(profile))
            if not passes_shape_screen(profile):
                raise ValueError("outside thickness or camber bounds")
            vector = _rms_vector(profile)
            distance, position = tree.query(vector)
            row.update(
                nearest_naca_id=naca_ids[position], nearest_naca_rms=float(distance)
            )
            if distance <= NEAR_DUPLICATE:
                raise ValueError("near duplicate of an AirfRANS NACA profile")
            if vectors:
                distances = np.linalg.norm(np.asarray(vectors) - vector, axis=1)
                nearest = int(distances.argmin())
                if distances[nearest] <= NEAR_DUPLICATE:
                    row["duplicate_of"] = retained[nearest][0]["filename"]
                    raise ValueError("near duplicate of a higher-ranked UIUC profile")
        except ValueError as error:
            row.update(status="excluded", reason=str(error))
        else:
            vectors.append(vector)
            retained.append((row, profile))
            row.update(status="eligible", rank=len(retained))
    audited.extend(
        {
            "filename": filename,
            "index_labels": sorted(labels[filename]),
            "status": "excluded",
            "reason": "listed in the index but absent from the coordinate archive",
        }
        for filename in sorted(labels.keys() - entries.keys())
    )
    return audited, retained


def _geometry_key(profile: Profile) -> bytes:
    return np.rint(profile.ordinate_samples * 1e12).astype("<i8").tobytes()


def rank_controls(
    splits: dict, profiles: dict[str, Profile]
) -> tuple[list[dict], list[dict]]:
    seen = {_geometry_key(profiles[case]) for case in splits["full_train"]}
    audited, retained = [], []
    for family, size in (("naca4", 3), ("naca5", 4)):
        rank = 0
        for case in sorted(splits["full_test"], key=selection_key):
            params = tuple(map(float, case.split("_")[4:]))
            if len(params) != size:
                continue
            profile = profiles[case]
            row = {
                "id": case,
                "family": family,
                "naca_parameters": params,
                "order_sha256": selection_key(case),
                **_shape_record(profile),
            }
            audited.append(row)
            key = _geometry_key(profile)
            if key in seen:
                row.update(status="excluded", reason="duplicate NACA geometry")
            elif not passes_shape_screen(profile):
                row.update(
                    status="excluded", reason="outside thickness or camber bounds"
                )
            else:
                rank += 1
                row.update(
                    status="eligible",
                    rank=rank,
                    role="primary" if rank <= 15 else "reserve",
                )
                retained.append(row)
                seen.add(key)
    return audited, retained


def prepare_cohorts(
    splits_path: Path, index_path: Path, archive_path: Path, pin: dict, out: Path
) -> None:
    if out.exists():
        raise FileExistsError(out)
    split_bytes = splits_path.read_bytes()
    if _sha256(split_bytes) != pin["split_sha256"]:
        raise ValueError("original AirfRANS split checksum mismatch")
    splits = json.loads(split_bytes)
    profiles = {}
    for case_id in splits["full_train"] + splits["full_test"]:
        shape = tuple(map(float, case_id.split("_")[4:]))
        contour = naca_generator(shape, nb_samples=2000, verbose=False, CTE=True)
        profiles[case_id] = align_profile(*split_contour(contour))

    index, raw_archive = index_path.read_bytes(), archive_path.read_bytes()
    with ZipFile(io.BytesIO(raw_archive)) as archive:
        uiuc_audit, retained = screen_uiuc(index, archive, profiles)
    naca_audit, controls = rank_controls(splits, profiles)
    out.mkdir(parents=True, exist_ok=False)
    (out / "uiuc-index.html").write_bytes(index)
    (out / "uiuc-coordinates.zip").write_bytes(raw_archive)
    sources = {
        "uiuc_index": {"url": UIUC_INDEX, "sha256": _sha256(index)},
        "uiuc_coordinates": {"url": UIUC_ARCHIVE, "sha256": _sha256(raw_archive)},
        "airfrans_split_sha256": _sha256(split_bytes),
    }
    (out / "screening.json").write_text(
        json.dumps(
            {"sources": sources, "uiuc": uiuc_audit, "naca": naca_audit},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if len(retained) < 30 or any(
        sum(row["family"] == family for row in controls) < 15
        for family in ("naca4", "naca5")
    ):
        raise ValueError("insufficient eligible geometries; see screening.json")

    geometry = out / "geometry"
    geometry.mkdir()
    conditions = {
        "inlet_velocity": 4e6 * AIRFRANS_VISCOSITY,
        "aoa_deg": 4.0,
        "reynolds": 4e6,
        "iteration_limit": 20_000,
    }
    cases = [row | conditions | {"group": "naca"} for row in controls]
    for row, profile in retained:
        case_id = "uiuc-" + Path(row["filename"]).stem
        path = geometry / f"{case_id}.csv"
        np.savetxt(path, profile.contour(), delimiter=",", fmt="%.17g")
        cases.append(
            row
            | conditions
            | {
                "id": case_id,
                "group": "uiuc",
                "role": "primary" if row["rank"] <= 30 else "reserve",
                "geometry_file": path.relative_to(out).as_posix(),
                "geometry_sha256": _sha256(path.read_bytes()),
            }
        )
    manifest = {
        "schema": 1,
        "generator_revision": GENERATOR_REVISION,
        "sources": sources,
        "cases": cases,
    }
    (out / "cohorts.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_splits", type=Path)
    parser.add_argument("uiuc_index", type=Path)
    parser.add_argument("uiuc_coordinates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare_cohorts(
        args.official_splits,
        args.uiuc_index,
        args.uiuc_coordinates,
        json.loads(PIN.read_text()),
        args.out,
    )
    print(args.out / "cohorts.json")


if __name__ == "__main__":
    main()
