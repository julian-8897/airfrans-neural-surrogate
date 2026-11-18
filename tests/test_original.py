import numpy as np
import pytest
from meshes import grid_with_hole
from numpy.testing import assert_allclose, assert_array_equal

pv = pytest.importorskip("pyvista")
original = pytest.importorskip("airfoil_surrogate.data.original")


@pytest.fixture
def mesh_files(tmp_path):
    pos, triangles, node_id = grid_with_hole({(2, 2)})
    points = np.column_stack((pos, np.full(len(pos), 0.5)))
    cells = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    internal = pv.UnstructuredGrid(
        cells, np.full(len(triangles), pv.CellType.TRIANGLE, dtype=np.uint8), points
    )
    wall = [node_id(3, 3), node_id(3, 2), node_id(2, 2), node_id(2, 3)]
    velocity = np.tile([2.0, 0.5, 0.0], (len(pos), 1))
    velocity[[0, *wall]] = 0.0
    internal.point_data["U"] = velocity
    internal.cell_data["U"] = np.tile([-99.0, 0.0, 0.0], (len(triangles), 1))
    internal.point_data["p"] = pos[:, 0]
    internal.point_data["nut"] = np.full(len(pos), 1e-5)
    internal.point_data["implicit_distance"] = -np.ones(len(pos))
    internal.point_data["implicit_distance"][wall] = 0.0

    patch = pv.PolyData(points[wall], lines=[2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
    outward = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]])
    patch.point_data["Normals"] = -outward / np.sqrt(2)
    patch.cell_data["Normals"] = np.array(
        [[-1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [0, -1.0, 0]]
    )
    internal_path = tmp_path / "internal.vtu"
    patch_path = tmp_path / "aerofoil.vtp"
    internal.save(internal_path)
    patch.save(patch_path)
    return internal_path, patch_path, internal, patch, wall, outward


def test_original_features_preserve_mesh_and_derive_wall_from_patch(mesh_files):
    internal_path, patch_path, internal, _, wall, outward = mesh_files
    mesh = original.load_original_mesh(internal_path, patch_path)
    features = mesh.features(inlet_velocity=4.0, aoa=np.pi / 2)

    assert_array_equal(mesh.internal.cells, internal.cells)
    assert_array_equal(mesh.wall_nodes, wall)
    assert_array_equal(features.pos, internal.points[:, :2])
    assert_array_equal(features.x[:, :2], features.pos)
    assert_allclose(
        features.x[:, 2:4], np.tile([0, 4], (len(features.pos), 1)), atol=1e-15
    )
    assert_array_equal(features.x[:, 4], (~features.surf).astype(float))
    assert_array_equal(np.flatnonzero(features.surf), sorted(wall))
    assert_array_equal(features.x[~features.surf, 5:7], 0)
    assert_allclose(features.x[wall, 5:7], outward[:, :2] / np.sqrt(2))
    assert_array_equal(features.y[:, :2], internal.point_data["U"][:, :2])
    assert_array_equal(features.y[:, 2], internal.point_data["p"])
    assert_array_equal(features.y[:, 3], internal.point_data["nut"])
    assert features.x.dtype == features.y.dtype == np.float64


def test_original_mesh_rejects_a_mismatched_wall_patch(mesh_files):
    internal_path, patch_path, _, patch, _, _ = mesh_files
    patch.points[0, 0] += 1e-6
    patch.save(patch_path)
    with pytest.raises(ValueError, match="wall patch must match"):
        original.load_original_mesh(internal_path, patch_path)


def test_original_force_adapter_on_known_fields(mesh_files):
    forces = pytest.importorskip("airfoil_surrogate.eval.forces")
    internal_path, patch_path, _, _, _, outward = mesh_files
    mesh = original.load_original_mesh(internal_path, patch_path)
    sim = forces.OriginalForceSimulation(mesh, inlet_velocity=2.0, aoa=0.0)
    reference = forces.force_coefficients(sim, reference=True)

    fields = np.zeros((len(sim.position), 4))
    fields[:, 2] = 17.0
    sim.set_fields(fields)
    assert_allclose(forces.force_coefficients(sim), 0, atol=1e-12)

    # AirfRANS averages the unit diagonal vertex normals onto each edge. Their
    # edge-average magnitudes are 1/sqrt(2), scaling the exact rectangular-wall
    # force of -0.5 for dp/dx = 1. The dynamic pressure at U = 2 is 2.
    fields[:, 2] += sim.position[:, 0]
    sim.set_fields(fields)
    drag = -0.25 / np.sqrt(2)
    assert_allclose(
        forces.force_coefficients(sim), [[0, 0, 0], [drag, drag, 0]], rtol=0, atol=1e-12
    )

    fields[:, 0] = sim.position[:, 1]
    sim.set_fields(fields)
    assert_allclose(
        sim.wallshearstress(over_airfoil=True),
        sim.NU * outward[:, [1, 0]] / np.sqrt(2),
        rtol=0,
        atol=np.finfo(np.float64).eps * sim.NU,
    )
    assert_array_equal(forces.force_coefficients(sim, reference=True), reference)
