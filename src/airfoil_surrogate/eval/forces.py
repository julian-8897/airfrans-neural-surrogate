"""AirfRANS force integration on the prototype triangular meshes."""

import numpy as np
import pyvista as pv
from airfrans import Simulation
from scipy.stats import spearmanr

from airfoil_surrogate.data.features import wall_normals
from airfoil_surrogate.data.remeshed import RemeshedSample, wall_edges


class RemeshedForceSimulation(Simulation):
    """Provide the mesh and fields required by Simulation's force methods."""

    def __init__(self, sample: RemeshedSample):
        self.position = sample.pos
        self.inlet_velocity = sample.inlet_velocity
        self.angle_of_attack = sample.aoa
        # Simulation's viscosity at its default 298.15 K. Density cancels in CL/CD.
        self.NU = np.array(1.5498148291427463e-5)
        self.RHO = 1.0
        self.sdf = -sample.implicit_distance[:, None]

        points = np.column_stack((self.position, np.zeros(len(self.position))))
        triangles = sample.triangles
        cells = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
        types = np.full(len(triangles), pv.CellType.TRIANGLE, dtype=np.uint8)
        self.internal = pv.UnstructuredGrid(cells, types, points)
        self.internal.point_data["U"] = np.column_stack(
            (sample.fields[:, :2], np.zeros(len(points)))
        )
        self.internal.point_data["p"] = sample.fields[:, 2]

        edges, opposite = wall_edges(triangles, self.position)
        wall, inverse, degree = np.unique(
            edges, return_inverse=True, return_counts=True
        )
        if not len(wall) or np.any(degree != 2):
            raise ValueError(f"sample {sample.index}: wall must have closed boundaries")
        self.surface = np.zeros(len(points), dtype=bool)
        self.surface[wall] = True
        self.airfoil_position = self.position[wall]
        # AirfRANS uses normals towards the solid at both vertices and edges.
        self.normals = -wall_normals(triangles, self.position)
        lines = np.column_stack((np.full(len(edges), 2), inverse.reshape(-1, 2)))
        self.airfoil = pv.PolyData(
            points[wall], lines=lines.ravel()
        ).compute_cell_sizes(area=False, volume=False, vertex_count=False)
        start, end = self.position[edges[:, 0]], self.position[edges[:, 1]]
        tangent = end - start
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        points_into_fluid = (
            np.einsum("ij,ij->i", normals, self.position[opposite] - start) > 0
        )
        normals[points_into_fluid] *= -1
        self.airfoil.cell_data["Normals"] = (
            normals / self.airfoil.cell_data["Length"][:, None]
        )
        self.set_fields(sample.fields)

    def set_fields(self, fields: np.ndarray) -> None:
        self.velocity = fields[:, :2]
        self.pressure = fields[:, 2:3]


def force_coefficients(sim: Simulation, *, reference: bool = False) -> np.ndarray:
    """Rows are CL and CD, columns are total, pressure and viscous contributions."""
    drag, lift = sim.force_coefficient(reference=reference)
    return np.asarray([lift, drag])


def coefficient_relative_error(
    prediction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    return np.divide(
        np.abs(prediction - reference),
        np.abs(reference),
        out=np.full_like(reference, np.nan),
        where=reference != 0,
    )


def coefficient_spearman(
    prediction: np.ndarray, reference: np.ndarray
) -> list[float | None]:
    return [
        float(spearmanr(pred, ref).statistic)
        if len(ref) > 1 and np.ptp(pred) > 0 and np.ptp(ref) > 0
        else None
        for pred, ref in zip(prediction.T, reference.T, strict=True)
    ]


def force_errors(
    sample: RemeshedSample, prediction: np.ndarray
) -> dict[str, np.ndarray]:
    """Compare predictions with ground truth integrated on the same mesh."""
    sim = RemeshedForceSimulation(sample)
    reference = force_coefficients(sim, reference=True)
    reference_wss = sim.wallshearstress(over_airfoil=True, reference=True)
    sim.set_fields(prediction)
    predicted = force_coefficients(sim)
    predicted_wss = sim.wallshearstress(over_airfoil=True)
    stored = np.array([sample.cl, sample.cd])
    return {
        "coefficients_predicted": predicted,
        "coefficients_reference": reference,
        "coefficients_stored": stored,
        "coefficients_relative_error": coefficient_relative_error(
            predicted[:, 0], reference[:, 0]
        ),
        "coefficients_stored_relative_error": coefficient_relative_error(
            predicted[:, 0], stored
        ),
        "coefficients_reference_minus_stored": reference[:, 0] - stored,
        "wss_rmse": np.sqrt(np.mean((predicted_wss - reference_wss) ** 2, axis=0)),
    }
