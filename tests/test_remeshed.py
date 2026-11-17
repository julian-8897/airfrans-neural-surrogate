import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from airfoil_surrogate.data.remeshed import (
    BOX,
    TASK_SPLITS,
    airfoil_boundary_nodes,
    case_key,
    iter_samples,
    load_split_indices,
    match_case_names,
    parse_sample,
    unpickle_sample,
)

DATA_DIR = Path(
    os.environ.get(
        "AIRFRANS_REMESHED_DIR",
        Path(__file__).resolve().parents[1] / "data/raw/airfrans_remeshed",
    )
)


def grid_with_hole(hole: set[tuple[int, int]]):
    """Triangulated grid over BOX with the quads in ``hole`` removed."""
    xs = np.linspace(BOX[0], BOX[1], 7)
    ys = np.linspace(BOX[2], BOX[3], 7)
    pos = np.array([(x, y) for y in ys for x in xs])
    nid = lambda i, j: j * len(xs) + i  # noqa: E731
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            if (i, j) in hole:
                continue
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    return pos, np.array(tris), nid


def cgns(name, value=None, children=(), label="DataArray_t"):
    return [name, value, list(children), label]


def raw_sample(conn: np.ndarray, n_nodes: int) -> dict:
    rng = np.random.default_rng(0)
    n_cells = len(conn) // 3
    zone_ids = np.arange(n_cells) % 6
    zone = cgns(
        "Zone",
        children=[
            cgns(
                "GridCoordinates",
                children=[
                    cgns("CoordinateX", rng.random(n_nodes)),
                    cgns("CoordinateY", rng.random(n_nodes)),
                ],
            ),
            cgns("Elem_Triangles", children=[cgns("ElementConnectivity", conn)]),
            cgns(
                "VertexFields",
                children=[
                    cgns(k, rng.random(n_nodes))
                    for k in ("implicit_distance", "nut", "p", "Ux", "Uy")
                ],
            ),
            cgns(
                "CellCenterFields",
                children=[
                    cgns(f"zone_{k}", (zone_ids == k).astype(float)) for k in range(6)
                ],
            ),
        ],
    )
    tree = cgns("CGNSTree", children=[cgns("Base_2_2", children=[zone])])
    scalars = {
        np.str_("C_D"): np.float64(0.01),
        np.str_("C_L"): np.float64(0.5),
        np.str_("angle_of_attack"): np.float64(0.1),
        np.str_("inlet_velocity"): np.float64(50.0),
    }
    return {"meshes": {0.0: tree}, "scalars": scalars}


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo unpickled",))


def test_unpickler_blocks_non_numpy_globals():
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        unpickle_sample(pickle.dumps(_Exploit()))


def test_unpickler_round_trips_numpy_payload():
    payload = {"a": np.arange(5.0), "s": np.float64(2.5)}
    out = unpickle_sample(pickle.dumps(payload))
    np.testing.assert_array_equal(out["a"], payload["a"])
    assert out["s"] == 2.5


@pytest.mark.parametrize(
    "hole, expected_nodes",
    [
        ({(2, 3)}, 4),  # single quad: its four corners
        ({(2, 2), (3, 2), (2, 3), (3, 3)}, 8),  # 2x2 quads: perimeter only
    ],
)
def test_airfoil_boundary_nodes_finds_hole_perimeter(hole, expected_nodes):
    pos, tris, nid = grid_with_hole(hole)
    nodes = airfoil_boundary_nodes(tris, pos)
    corners = {nid(i + di, j + dj) for i, j in hole for di in (0, 1) for dj in (0, 1)}
    assert len(nodes) == expected_nodes
    assert set(nodes) <= corners


def test_airfoil_boundary_nodes_empty_without_hole():
    pos, tris, _ = grid_with_hole(set())
    assert airfoil_boundary_nodes(tris, pos).size == 0


def test_parse_sample_converts_to_zero_based():
    conn = np.array([1, 2, 3, 1, 3, 4], dtype=np.int64)
    s = parse_sample(7, raw_sample(conn, 4))
    np.testing.assert_array_equal(s.triangles, [[0, 1, 2], [0, 2, 3]])
    assert s.fields.shape == (4, 4) and s.pos.shape == (4, 2)
    assert s.cell_zone.tolist() == [0, 1]
    assert (s.index, s.cl, s.cd) == (7, 0.5, 0.01)


def test_parse_sample_rejects_zero_based_connectivity():
    conn = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="1-based"):
        parse_sample(0, raw_sample(conn, 4))


def test_case_key_matches_airfrans_name_format():
    # airFoil2D_SST_31.283_-4.156_... : U = 31.283 m/s, AoA = -4.156 deg
    assert case_key(31.283, np.radians(-4.156)) == (31.283, -4.156)


requires_data = pytest.mark.skipif(
    not DATA_DIR.exists(), reason=f"AirfRANS_remeshed not found at {DATA_DIR}"
)


@pytest.fixture(scope="module")
def dataset_summary():
    rows = []
    for s in iter_samples(DATA_DIR):
        wall = airfoil_boundary_nodes(s.triangles, s.pos)
        rows.append(
            {
                "key": case_key(s.inlet_velocity, s.aoa),
                "n_wall": len(wall),
                "wall_dist": float(np.abs(s.implicit_distance[wall]).max()),
                "in_box": bool(
                    (s.pos[:, 0] >= BOX[0]).all()
                    and (s.pos[:, 0] <= BOX[1]).all()
                    and (np.abs(s.pos[:, 1]) <= BOX[3]).all()
                ),
            }
        )
    return rows


@requires_data
@pytest.mark.slow
def test_all_samples_parse_and_lie_in_box(dataset_summary):
    assert len(dataset_summary) == 1000
    assert all(r["in_box"] for r in dataset_summary)


@requires_data
@pytest.mark.slow
def test_wall_nodes_have_near_zero_distance(dataset_summary):
    # Chord length is 1; the largest measured wall |d| is 3.2e-4 chords.
    assert all(r["n_wall"] > 0 for r in dataset_summary)
    assert max(r["wall_dist"] for r in dataset_summary) < 1e-3


@requires_data
def test_splits_match_airfrans_task_sizes_and_are_disjoint():
    splits = load_split_indices(DATA_DIR / "README.md")
    sizes = {k: len(splits[k]) for k in TASK_SPLITS}
    assert sizes == {
        "full_train": 800,
        "full_test": 200,
        "scarce_train": 200,
        "reynolds_train": 504,
        "reynolds_test": 496,
        "aoa_train": 804,
        "aoa_test": 196,
    }
    for task in ("full", "reynolds", "aoa"):
        train, test = set(splits[f"{task}_train"]), set(splits[f"{task}_test"])
        assert not train & test
        assert train | test == set(range(1000))
    assert set(splits["scarce_train"]) <= set(splits["full_train"])


@requires_data
@pytest.mark.slow
def test_case_names_match_official_manifest_splits(dataset_summary):
    import json

    manifest_path = DATA_DIR / "airfrans_manifest.json"
    if not manifest_path.exists():
        pytest.skip("run scripts/download_remeshed.py to fetch the case manifest")
    names = match_case_names([r["key"] for r in dataset_summary], manifest_path)
    assert len(set(names)) == 1000
    manifest = json.loads(manifest_path.read_text())
    splits = load_split_indices(DATA_DIR / "README.md")
    for split in TASK_SPLITS:
        assert {names[i] for i in splits[split]} == set(manifest[split])
