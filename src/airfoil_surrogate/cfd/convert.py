"""Convert one OpenFOAM VTK export to the processed AirfRANS mesh layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyvista as pv

from airfoil_surrogate.data import BOX
from airfoil_surrogate.data.original import OriginalMesh, match_original_mesh

CLIP_BOX = (*BOX, 0.0, 1.0)
SLICE_Z = 0.5
BASE_FIELDS = ("U", "p", "nut")


def _keep_fields(mesh: pv.DataSet, fields: tuple[str, ...]) -> None:
    for data in (mesh.point_data, mesh.cell_data):
        for key in list(data.keys()):
            if key not in fields:
                del data[key]


def convert_export(
    volume: pv.UnstructuredGrid, wall: pv.PolyData
) -> tuple[OriginalMesh, dict]:
    """Crinkle-clip, slice and qualify a volume and its oriented aerofoil patch."""
    if not isinstance(volume, pv.UnstructuredGrid) or not isinstance(wall, pv.PolyData):
        raise ValueError("expected a volume VTU and a wall VTP from foamToVTK")
    if not wall.n_faces or not volume.n_cells:
        raise ValueError("the VTK volume and wall surface must have cells")

    oriented_wall = wall.compute_normals(
        point_normals=True,
        cell_normals=True,
        flip_normals=False,
        auto_orient_normals=False,
    )
    slice_options = {
        "normal": "z",
        "origin": (0.0, 0.0, SLICE_Z),
        "generate_triangles": False,
    }
    internal = volume.clip_box(CLIP_BOX, invert=False, crinkle=True).slice(
        **slice_options
    )
    internal = internal.cast_to_unstructured_grid()
    internal.compute_implicit_distance(oriented_wall, inplace=True)
    aerofoil = oriented_wall.slice(**slice_options)
    _keep_fields(internal, (*BASE_FIELDS, "implicit_distance"))
    _keep_fields(aerofoil, (*BASE_FIELDS, "Normals"))
    mesh = match_original_mesh(internal, aerofoil)

    # Validate the same arrays and wall mask used by the field model.
    features = mesh.features(inlet_velocity=1.0, aoa=0.0)
    wall_spacing = np.abs(features.x[features.surf, 4]).max()
    if wall_spacing > 1e-7 or np.any(features.x[~features.surf, 4] <= 0):
        raise ValueError("distance sign or wall alignment differs from AirfRANS")
    for data in (aerofoil.point_data, aerofoil.cell_data):
        normals = np.asarray(data["Normals"], dtype=np.float64)
        if not np.allclose(
            np.linalg.norm(normals[:, :2], axis=1), 1.0, rtol=0, atol=1e-6
        ):
            raise ValueError("expected unit planar wall normals")

    offsets = {}
    for field in BASE_FIELDS:
        boundary = np.asarray(aerofoil.point_data[field], dtype=np.float64)
        interior = np.asarray(
            internal.point_data[field][mesh.wall_nodes], dtype=np.float64
        )
        if boundary.shape != interior.shape or not np.isfinite(boundary).all():
            raise ValueError(f"invalid wall point field {field!r}")
        offsets[field] = float(np.abs(interior - boundary).max())
    speed = float(np.linalg.norm(features.y[features.surf, :2], axis=1).max())
    boundary_speed = float(np.linalg.norm(aerofoil.point_data["U"], axis=1).max())
    if max(speed, boundary_speed) > 1e-7:
        raise ValueError("converted aerofoil wall does not satisfy no-slip")

    return mesh, {
        "nodes": internal.n_points,
        "cells": internal.n_cells,
        "wall_nodes": aerofoil.n_points,
        "max_abs_wall_distance": float(wall_spacing),
        "max_wall_speed": speed,
        "max_patch_wall_speed": boundary_speed,
        "max_abs_patch_minus_internal_at_wall": offsets,
    }


def _file_record(path: Path) -> dict:
    with path.open("rb") as stream:
        return {
            "size": path.stat().st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def convert_files(internal: Path, aerofoil: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    sources = {
        "internal.vtu": _file_record(internal),
        "aerofoil.vtp": _file_record(aerofoil),
    }
    mesh, diagnostics = convert_export(pv.read(internal), pv.read(aerofoil))
    out.mkdir(parents=True, exist_ok=False)
    mesh.internal.save(out / "internal.vtu")
    mesh.aerofoil.save(out / "aerofoil.vtp")
    with (out / "conversion.json").open("x") as stream:
        json.dump(
            {
                "schema": 1,
                "source": sources,
                "output": {name: _file_record(out / name) for name in sources},
                "clip_box": CLIP_BOX,
                "slice_z": SLICE_Z,
                "versions": {name: version(name) for name in ("pyvista", "vtk")},
                "diagnostics": diagnostics,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal", type=Path, help="3D foamToVTK volume mesh")
    parser.add_argument("aerofoil", type=Path, help="3D foamToVTK aerofoil patch")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    convert_files(args.internal, args.aerofoil, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
