"""Read original AirfRANS meshes without changing their node or cell order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from airfoil_surrogate.data.features import NodeFeatures
from airfoil_surrogate.data.staged_original import verified_mesh_paths


@dataclass(frozen=True)
class OriginalMesh:
    internal: pv.UnstructuredGrid
    aerofoil: pv.PolyData
    wall_nodes: np.ndarray  # Internal node IDs in aerofoil patch order.

    def features(self, inlet_velocity: float, aoa: float) -> NodeFeatures:
        """Build dimensional features and targets, taking the inlet angle in radians."""
        if not np.isfinite([inlet_velocity, aoa]).all() or inlet_velocity <= 0:
            raise ValueError("inlet speed must be positive and inlet conditions finite")

        pos = np.asarray(self.internal.points[:, :2], dtype=np.float64)
        normals = np.zeros_like(pos)
        inward = _point_array(self.aerofoil, "Normals", 3)
        normals[self.wall_nodes] = -inward[:, :2]
        surf = np.zeros(len(pos), dtype=bool)
        surf[self.wall_nodes] = True

        u_inf = inlet_velocity * np.array([np.cos(aoa), np.sin(aoa)])
        distance = _point_array(self.internal, "implicit_distance")
        velocity = _point_array(self.internal, "U", 3)
        pressure = _point_array(self.internal, "p")
        nu_t = _point_array(self.internal, "nut")
        x = np.column_stack(
            (pos, np.broadcast_to(u_inf, pos.shape), -distance, normals)
        )
        y = np.column_stack((velocity[:, :2], pressure, nu_t))
        return NodeFeatures(pos=pos, x=x, y=y, surf=surf)


def _point_array(mesh: pv.DataSet, name: str, components: int = 1) -> np.ndarray:
    data = np.asarray(mesh.point_data[name], dtype=np.float64)
    expected = (mesh.n_points,) if components == 1 else (mesh.n_points, components)
    if data.shape != expected or not np.isfinite(data).all():
        raise ValueError(f"invalid vertex field {name!r}: expected finite {expected}")
    return data


def load_original_mesh(internal_path: Path, aerofoil_path: Path) -> OriginalMesh:
    """Read a processed VTU/VTP pair and match the wall by exact planar coordinates."""
    internal = pv.read(internal_path)
    aerofoil = pv.read(aerofoil_path)
    if not isinstance(internal, pv.UnstructuredGrid) or not isinstance(
        aerofoil, pv.PolyData
    ):
        raise ValueError(
            "expected an internal unstructured grid and a wall polydata patch"
        )
    if (
        not internal.n_cells
        or not aerofoil.n_lines
        or not all(np.isfinite(mesh.points).all() for mesh in (internal, aerofoil))
    ):
        raise ValueError("expected nonempty original meshes with finite coordinates")

    # The AirfRANS force routine also requires an exact 2D match. Check the
    # second neighbour to reject coincident internal vertices at the wall.
    distance, neighbours = cKDTree(internal.points[:, :2]).query(
        aerofoil.points[:, :2], k=2
    )
    wall_nodes = neighbours[:, 0]
    if (
        np.any(distance[:, 0] != 0)
        or np.any(distance[:, 1] == 0)
        or np.unique(wall_nodes).size != aerofoil.n_points
        or aerofoil.n_points >= internal.n_points
    ):
        raise ValueError("wall patch must match a unique subset of internal vertices")
    return OriginalMesh(internal, aerofoil, wall_nodes)


def load_staged_features(root: Path, row: dict) -> NodeFeatures:
    """Verify a staged case's files and build its original-node features."""
    inlet_velocity, angle_deg = map(float, row["id"].split("_")[2:4])
    mesh = load_original_mesh(*verified_mesh_paths(root, row))
    return mesh.features(inlet_velocity, np.deg2rad(angle_deg))
