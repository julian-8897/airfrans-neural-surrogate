"""Loader for the PLAID ``AirfRANS_remeshed`` dataset (prototype data).

Each Parquet row holds one pickled PLAID sample: a dict with a CGNS mesh tree
(triangular mesh clipped to ``BOX`` and coarsened with MMG) and four scalars.
Samples are unpickled with a numpy-only allow-list, so a tampered file cannot
import arbitrary code. Row order across the sorted shards defines the sample
index used by the split lists in the dataset card.
"""

from __future__ import annotations

import io
import json
import pickle
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

FIELD_NAMES = ("Ux", "Uy", "p", "nut")
N_ZONES = 6
BOX = (-2.0, 4.0, -1.5, 1.5)  # xmin, xmax, ymin, ymax of the clipped domain
TASK_SPLITS = (
    "full_train",
    "full_test",
    "scarce_train",
    "reynolds_train",
    "reynolds_test",
    "aoa_train",
    "aoa_test",
)

_ALLOWED_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")
        # Pickles written with numpy 1.x name numpy.core; numpy 2 moved it.
        return super().find_class(module.replace("numpy.core", "numpy._core"), name)


def unpickle_sample(blob: bytes) -> dict:
    """Unpickle one sample, refusing any global outside numpy arrays."""
    return _NumpyOnlyUnpickler(io.BytesIO(blob)).load()


@dataclass(frozen=True)
class RemeshedSample:
    index: int
    pos: np.ndarray  # (N, 2) float64 node coordinates
    triangles: np.ndarray  # (M, 3) int64, zero-based node indices
    fields: np.ndarray  # (N, 4) float64, columns FIELD_NAMES
    implicit_distance: np.ndarray  # (N,) float64, negative in the fluid
    cell_zone: np.ndarray  # (M,) int8, index of the zone_k indicator set
    inlet_velocity: float  # m/s
    aoa: float  # rad
    cl: float
    cd: float


def _child(node: list, name: str) -> list:
    for child in node[2]:
        if child[0] == name:
            return child
    raise KeyError(f"CGNS node {node[0]!r} has no child {name!r}")


def _array(node: list, *path: str) -> np.ndarray:
    for name in path:
        node = _child(node, name)
    return node[1]


def parse_sample(index: int, raw: dict) -> RemeshedSample:
    """Convert an unpickled PLAID sample dict into a validated RemeshedSample."""
    tree = next(iter(raw["meshes"].values()))
    zone = _child(_child(tree, "Base_2_2"), "Zone")
    pos = np.stack(
        [
            _array(zone, "GridCoordinates", "CoordinateX"),
            _array(zone, "GridCoordinates", "CoordinateY"),
        ],
        axis=1,
    ).astype(np.float64)
    n = len(pos)

    conn = _array(zone, "Elem_Triangles", "ElementConnectivity").astype(np.int64)
    if conn.size % 3 or conn.min() < 1 or conn.max() > n:
        raise ValueError(f"sample {index}: connectivity is not 1-based triangles")
    triangles = conn.reshape(-1, 3) - 1

    fields = np.stack([_array(zone, "VertexFields", k) for k in FIELD_NAMES], axis=1)
    distance = _array(zone, "VertexFields", "implicit_distance")
    indicators = np.stack(
        [_array(zone, "CellCenterFields", f"zone_{k}") for k in range(N_ZONES)],
        axis=1,
    )
    if not np.all(indicators.sum(axis=1) == 1):
        raise ValueError(f"sample {index}: zone indicators are not a partition")

    scalars = {str(k): float(v) for k, v in raw["scalars"].items()}
    sample = RemeshedSample(
        index=index,
        pos=pos,
        triangles=triangles,
        fields=fields.astype(np.float64),
        implicit_distance=distance.astype(np.float64),
        cell_zone=indicators.argmax(axis=1).astype(np.int8),
        inlet_velocity=scalars["inlet_velocity"],
        aoa=scalars["angle_of_attack"],
        cl=scalars["C_L"],
        cd=scalars["C_D"],
    )
    if sample.fields.shape != (n, len(FIELD_NAMES)) or distance.shape != (n,):
        raise ValueError(f"sample {index}: vertex field shapes do not match mesh")
    if not all(np.isfinite(a).all() for a in (pos, sample.fields, distance)):
        raise ValueError(f"sample {index}: non-finite values")
    return sample


def shard_paths(data_dir: Path) -> list[Path]:
    paths = sorted(Path(data_dir).glob("data/all_samples-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet shards under {data_dir}/data")
    return paths


def iter_samples(
    data_dir: Path, indices: Iterable[int] | None = None
) -> Iterator[RemeshedSample]:
    """Yield samples in ascending index order, optionally restricted to ``indices``.

    The order of ``indices`` is ignored; use ``RemeshedSample.index`` to align.
    """
    wanted = None if indices is None else set(indices)
    offset = 0
    for path in shard_paths(data_dir):
        column = pq.read_table(path, columns=["sample"]).column("sample")
        for i in range(len(column)):
            index = offset + i
            if wanted is None or index in wanted:
                blob = column[i].as_py()
                yield parse_sample(index, unpickle_sample(blob))
        offset += len(column)


def load_split_indices(card_path: Path) -> dict[str, np.ndarray]:
    """Read the split index lists from the dataset card's YAML header."""
    text = Path(card_path).read_text()
    header = text.split("---")[1]
    splits = yaml.safe_load(header)["dataset_info"]["description"]["split"]
    return {name: np.asarray(ids, dtype=np.int64) for name, ids in splits.items()}


def case_key(inlet_velocity: float, aoa: float) -> tuple[float, float]:
    """Key used in AirfRANS case names: U to 3 d.p. and AoA in degrees to 3 d.p."""
    return round(inlet_velocity, 3), round(float(np.degrees(aoa)), 3)


def match_case_names(
    keys: Iterable[tuple[float, float]], manifest_path: Path
) -> list[str]:
    """Map (U, AoA) keys to original AirfRANS case names via manifest.json.

    Names have the form ``airFoil2D_SST_<U>_<AoA deg>_<NACA params...>``.
    Raises if any key is missing or ambiguous.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    names = set(manifest["full_train"]) | set(manifest["full_test"])
    lookup: dict[tuple[float, float], list[str]] = {}
    for name in names:
        u, a = name.split("_")[2:4]
        lookup.setdefault((float(u), float(a)), []).append(name)
    out = []
    for key in keys:
        found = lookup.get(key, [])
        if len(found) != 1:
            raise KeyError(f"{len(found)} AirfRANS cases match key {key}")
        out.append(found[0])
    return out


def wall_edges(
    triangles: np.ndarray, pos: np.ndarray, box=BOX, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Airfoil wall edges (E, 2) and, for each, the opposite vertex of its triangle.

    A boundary edge belongs to exactly one triangle; edges not on the outer
    clip box are on the airfoil.
    """
    local = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])  # (edge a, edge b, opposite)
    tri_edges = triangles[:, local].reshape(-1, 3)
    key = np.sort(tri_edges[:, :2], axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = tri_edges[counts[inverse.ravel()] == 1]
    xmin, xmax, ymin, ymax = box
    px, py = pos[:, 0], pos[:, 1]
    on_box = (
        (np.abs(px - xmin) < tol)
        | (np.abs(px - xmax) < tol)
        | (np.abs(py - ymin) < tol)
        | (np.abs(py - ymax) < tol)
    )
    airfoil = ~(on_box[boundary[:, 0]] & on_box[boundary[:, 1]])
    return boundary[airfoil, :2], boundary[airfoil, 2]


def airfoil_boundary_nodes(
    triangles: np.ndarray, pos: np.ndarray, box=BOX, tol: float = 1e-9
) -> np.ndarray:
    """Sorted node indices on the airfoil wall."""
    return np.unique(wall_edges(triangles, pos, box, tol)[0])
