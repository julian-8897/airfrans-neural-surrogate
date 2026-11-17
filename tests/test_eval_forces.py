from dataclasses import replace

import numpy as np
import pytest
from meshes import grid_with_hole
from numpy.testing import assert_allclose

from airfoil_surrogate.data.remeshed import RemeshedSample

forces = pytest.importorskip("airfoil_surrogate.eval.forces")


def test_pressure_force_on_closed_rectangular_wall():
    pos, triangles, nid = grid_with_hole({(2, 2), (3, 2)})
    fields = np.zeros((len(pos), 4))
    fields[:, :2] = [2.0, -1.0]
    fields[:, 2] = 17.0
    sample = RemeshedSample(
        index=0,
        pos=pos,
        triangles=triangles,
        fields=fields,
        implicit_distance=np.zeros(len(pos)),
        cell_zone=np.zeros(len(triangles), dtype=np.int8),
        inlet_velocity=2.0,
        aoa=0.0,
        cl=0.0,
        cd=0.0,
    )
    sim = forces.RemeshedForceSimulation(sample)
    assert_allclose(forces.force_coefficients(sim, reference=True), 0, atol=1e-12)

    # The hole has unit area. A unit positive x-pressure gradient gives -1 Fx.
    linear_pressure = fields.copy()
    linear_pressure[:, 2] += pos[:, 0]
    sim.set_fields(linear_pressure)
    expected = np.array([[0.0, 0.0, 0.0], [-0.5, -0.5, 0.0]])
    assert_allclose(forces.force_coefficients(sim), expected, atol=1e-12)

    rotated = forces.RemeshedForceSimulation(
        replace(sample, fields=linear_pressure, aoa=np.pi / 2)
    )
    assert_allclose(
        forces.force_coefficients(rotated, reference=True), -expected[::-1], atol=1e-12
    )

    # Ux = y gives a constant shear rate and opposite traction on horizontal faces.
    fields[:, 0] = pos[:, 1]
    sim.set_fields(fields)
    wall_shear = sim.wallshearstress()
    assert_allclose(
        wall_shear[[nid(3, 2), nid(3, 3)]],
        [[-sim.NU, 0], [sim.NU, 0]],
        rtol=0,
        atol=np.finfo(np.float64).eps * sim.NU,
    )


def test_coefficient_statistics_with_ties_and_undefined_cases():
    truth = np.array([[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    prediction = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
    assert_allclose(
        forces.coefficient_relative_error(prediction, truth),
        [[0.5, np.nan], [1, np.nan], [1, np.nan]],
    )
    rho = forces.coefficient_spearman(prediction, truth)
    assert rho[0] == pytest.approx(-1.0)
    assert rho[1] is None
