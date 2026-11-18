import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pv = pytest.importorskip("pyvista")
conversion = pytest.importorskip("airfoil_surrogate.cfd.convert")
original = pytest.importorskip("airfoil_surrogate.data.original")


@pytest.fixture
def extruded_mesh():
    xs = [-5, -3, -1, 0, 1, 3, 5, 7]
    ys = [-4, -2, -0.25, 0.25, 2, 4]
    xy = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    count = len(xy)
    points = np.column_stack((np.tile(xy, (2, 1)), np.repeat([0.0, 1.0], count)))

    def node(i, j):
        return j * len(xs) + i

    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if (i, j) == (3, 2):
                continue
            bottom = np.array(
                [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            )
            cells.extend([8, *bottom, *(bottom + count)])
    volume = pv.UnstructuredGrid(
        np.array(cells),
        np.full(len(cells) // 9, pv.CellType.HEXAHEDRON, dtype=np.uint8),
        points,
    )
    loop = [node(3, 2), node(4, 2), node(4, 3), node(3, 3)]
    wall_ids = np.array([*loop, *(index + count for index in loop)])
    faces = [4, 0, 4, 5, 1, 4, 1, 5, 6, 2, 4, 2, 6, 7, 3, 4, 3, 7, 4, 0]
    wall = pv.PolyData(points[wall_ids], faces=faces)

    velocity = np.tile([2.0, 0.0, 0.0], (len(points), 1))
    velocity[wall_ids] = 0.0
    fields = {
        "U": velocity,
        "p": points @ np.array([1, 2, 1]),
        "nut": np.full(len(points), 1e-4),
        "drop_me": np.ones(len(points)),
    }
    for name, values in fields.items():
        volume.point_data[name] = values
        wall.point_data[name] = values[wall_ids]
    return volume, wall


def test_cfd_converter_preserves_crinkle_quads_and_original_force_layout(
    extruded_mesh, tmp_path
):
    forces = pytest.importorskip("airfoil_surrogate.eval.forces")
    volume, wall = extruded_mesh
    volume_path = tmp_path / "volume.vtu"
    wall_path = tmp_path / "wall.vtp"
    volume.save(volume_path)
    wall.save(wall_path)
    out = tmp_path / "converted"
    conversion.convert_files(volume_path, wall_path, out)
    mesh = original.load_original_mesh(out / "internal.vtu", out / "aerofoil.vtp")
    features = mesh.features(2.0, 0.0)
    assert mesh.internal.n_cells == 14
    assert_array_equal(mesh.internal.celltypes, pv.CellType.QUAD)
    assert_array_equal(mesh.internal.points[:, 2], 0.5)
    assert mesh.internal.bounds[:4] == (-3.0, 5.0, -2.0, 2.0)
    assert features.surf.sum() == 4
    assert_array_equal(features.y[features.surf, :2], 0)
    x, y = features.pos.T
    assert_allclose(features.y[:, 2], x + 2 * y + 0.5, rtol=0, atol=1e-12)
    distance = np.hypot(
        np.maximum(np.maximum(-x, x - 1), 0), np.maximum(np.abs(y) - 0.25, 0)
    )
    assert_allclose(features.x[:, 4], distance, rtol=0, atol=1e-7)
    assert set(mesh.internal.point_data) == {"U", "p", "nut", "implicit_distance"}

    sim = forces.OriginalForceSimulation(mesh, 2.0, 0.0)
    pressure_coefficients = forces.force_coefficients(sim, reference=True)[:, 1]
    assert_allclose(
        pressure_coefficients, np.array([-0.5, -0.25]) / np.sqrt(2), atol=1e-7
    )

    record = json.loads((out / "conversion.json").read_text())
    assert record["diagnostics"]["max_wall_speed"] == 0
    assert record["diagnostics"]["max_abs_patch_minus_internal_at_wall"] == {
        "U": 0,
        "p": 0,
        "nut": 0,
    }
    for name in ("internal.vtu", "aerofoil.vtp"):
        assert (
            record["output"][name]["sha256"]
            == hashlib.sha256((out / name).read_bytes()).hexdigest()
        )
    with pytest.raises(FileExistsError):
        conversion.convert_files(volume_path, wall_path, out)


def test_cfd_converter_rejects_a_wall_pointing_out_of_the_solid(extruded_mesh):
    volume, wall = extruded_mesh
    flipped = wall.copy()
    flipped.faces = wall.faces.reshape(-1, 5)[:, [0, 4, 3, 2, 1]].ravel()
    with pytest.raises(ValueError, match="distance sign"):
        conversion.convert_export(volume, flipped)
