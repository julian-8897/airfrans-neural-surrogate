"""Qualify native inputs and force integration on a fixed training-only audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from airfrans import Simulation

from airfoil_surrogate.data.original import OriginalMesh, load_original_mesh
from airfoil_surrogate.data.remeshed import (
    case_key,
    iter_samples,
    load_split_indices,
    match_case_names,
    shard_paths,
)
from airfoil_surrogate.data.stage_original import PIN
from airfoil_surrogate.data.staged_original import (
    read_complete_partition,
    verified_mesh_paths,
)
from airfoil_surrogate.eval.forces import OriginalForceSimulation, force_coefficients
from airfoil_surrogate.train import git_state

PREVIOUS_AUDIT = (0, 194, 437)
ADAPTER_ATOL = 1e-8
STORED_ATOL = 1e-5
REMESHED_PIN = PIN.with_name("airfrans_remeshed.json")


def select_audit(train_ids: list[str], label_dir: Path, pin: dict) -> list[dict]:
    """Take the original three audit cases, then the first nine others."""
    card = label_dir / "README.md"
    shards = shard_paths(label_dir)
    manifest = label_dir / pin["case_names"]["local_name"]
    files = {path.relative_to(label_dir).as_posix(): path for path in (card, *shards)}
    if files.keys() != pin["files"].keys():
        raise ValueError("stored force label files differ from the pinned snapshot")
    for path, record in [
        *((path, pin["files"][name]) for name, path in files.items()),
        (manifest, pin["case_names"]),
    ]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"] or path.stat().st_size != record["size"]:
            raise ValueError(f"stored force label checksum mismatch: {path}")

    indices = load_split_indices(card)["full_train"][: len(train_ids)].tolist()
    if not set(PREVIOUS_AUDIT) <= set(indices) or len(indices) != len(train_ids):
        raise ValueError("previous force audit cases must belong to the training stage")
    chosen = [
        *PREVIOUS_AUDIT,
        *[idx for idx in indices if idx not in PREVIOUS_AUDIT][:9],
    ]
    samples = list(iter_samples(label_dir, chosen))
    cases = match_case_names(
        [case_key(sample.inlet_velocity, sample.aoa) for sample in samples], manifest
    )
    by_index = {
        sample.index: (case, sample)
        for case, sample in zip(cases, samples, strict=True)
    }
    train_by_index = dict(zip(indices, train_ids, strict=True))
    if len(chosen) != 12 or any(
        index not in by_index or by_index[index][0] != train_by_index[index]
        for index in chosen
    ):
        raise ValueError("stored force cases do not match the pinned training order")
    result = []
    for index in chosen:
        case, sample = by_index[index]
        if not np.isfinite([sample.cl, sample.cd]).all():
            raise ValueError(f"non-finite stored force labels for {case}")
        result.append({"case": case, "index": index, "stored": [sample.cl, sample.cd]})
    return result


def _wall_cells(mesh: OriginalMesh, surf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the one fluid cell adjacent to each segment, in patch order."""
    patch = mesh.aerofoil
    lines = patch.lines
    if (
        patch.n_cells != patch.n_lines
        or len(lines) != 3 * patch.n_lines
        or np.any(lines[::3] != 2)
    ):
        raise ValueError("expected two-point wall segments")
    endpoints = lines.reshape(-1, 3)[:, 1:]
    grid = mesh.internal
    conn, offsets = grid.cell_connectivity, grid.cell_offsets
    next_node = np.roll(conn, -1)
    next_node[offsets[1:] - 1] = conn[offsets[:-1]]
    wall_edges = np.flatnonzero(surf[conn] & surf[next_node])
    first, second = conn[wall_edges], next_node[wall_edges]
    edge_keys = np.minimum(first, second) * grid.n_points + np.maximum(first, second)
    first, second = mesh.wall_nodes[endpoints].T
    keys = np.minimum(first, second) * grid.n_points + np.maximum(first, second)
    unique, inverse, counts = np.unique(
        edge_keys, return_inverse=True, return_counts=True
    )
    hits = unique.searchsorted(keys)
    if (
        len(unique) == 0
        or np.any(hits >= len(unique))
        or np.any(unique[hits] != keys)
        or np.any(counts[hits] != 1)
        or len(np.unique(keys)) != len(keys)
    ):
        raise ValueError("wall segments must each have one adjacent fluid cell")
    owner = np.empty(len(unique), dtype=np.int64)
    owner[inverse] = np.searchsorted(offsets[1:], wall_edges, side="right")
    return endpoints, owner[hits]


def input_checks(mesh: OriginalMesh, sim: OriginalForceSimulation) -> dict:
    """Check that wall normals point towards cells incident to the wall."""
    features = sim.features
    surf = features.surf
    endpoints, owner = _wall_cells(mesh, surf)
    patch = mesh.aerofoil
    point_normals = features.x[mesh.wall_nodes, 5:7]
    segment_normals = -np.asarray(patch.cell_data["Normals"][:, :2], dtype=np.float64)
    midpoint = features.pos[mesh.wall_nodes[endpoints]].mean(axis=1)
    interior = mesh.internal.cell_centers().points[owner, :2] - midpoint
    segment_alignment = np.einsum("ij,ij->i", segment_normals, interior)
    incident_normals = np.zeros_like(point_normals)
    np.add.at(
        incident_normals, endpoints.ravel(), np.repeat(segment_normals, 2, axis=0)
    )
    point_alignment = np.einsum("ij,ij->i", point_normals, incident_normals)
    if not np.isfinite(segment_normals).all():
        raise ValueError("wall segment normals must be finite")
    measurements = {
        "wall_nodes": int(surf.sum()),
        "max_wall_speed": float(np.linalg.norm(features.y[surf, :2], axis=1).max()),
        "max_wall_distance": float(np.abs(features.x[surf, 4]).max()),
        "min_off_wall_distance": float(features.x[~surf, 4].min()),
        "max_off_wall_normal": float(np.abs(features.x[~surf, 5:7]).max()),
        "max_normal_length_error": float(
            max(
                np.abs(np.linalg.norm(point_normals, axis=1) - 1.0).max(),
                np.abs(np.linalg.norm(segment_normals, axis=1) - 1.0).max(),
            )
        ),
        "min_segment_alignment": float(segment_alignment.min()),
        "min_point_alignment": float(point_alignment.min()),
    }
    return measurements | {
        "passed": bool(
            measurements["max_wall_speed"] <= 1e-7
            and measurements["max_wall_distance"] <= 1e-7
            and measurements["min_off_wall_distance"] > 0
            and measurements["max_off_wall_normal"] == 0
            and measurements["max_normal_length_error"] <= 1e-6
            and measurements["min_segment_alignment"] > 0
            and measurements["min_point_alignment"] > 0
        )
    }


def compare_forces(adapter: np.ndarray, direct: np.ndarray, stored: np.ndarray) -> dict:
    """Record all total, pressure and viscous discrepancies and both force gates."""
    if not all(np.isfinite(value).all() for value in (adapter, direct, stored)):
        raise ValueError("force audit found non-finite coefficients")
    adapter_delta = adapter - direct
    stored_delta = direct[:, 0] - stored
    adapter_passed = bool(np.all(np.abs(adapter_delta) <= ADAPTER_ATOL))
    stored_passed = bool(np.all(np.abs(stored_delta) <= STORED_ATOL))
    return {
        "adapter": adapter.tolist(),
        "direct": direct.tolist(),
        "adapter_minus_direct": adapter_delta.tolist(),
        "direct_minus_stored": stored_delta.tolist(),
        "adapter_passed": adapter_passed,
        "stored_label_passed": stored_passed,
        "passed": adapter_passed and stored_passed,
    }


def qualify(
    train: Path, label_dir: Path, pin: dict, remeshed_pin: dict, out: Path
) -> bool:
    """Audit original training meshes against direct AirfRANS and stored labels."""
    if out.exists():
        raise FileExistsError(out)
    if remeshed_pin["case_names"]["sha256"] != pin["split_sha256"]:
        raise ValueError(
            "stored labels and original meshes must use the same case manifest"
        )
    rows, stage_digest = read_complete_partition(train, "train", pin)
    row_by_id = {row["id"]: row for row in rows}
    audit = select_audit(list(row_by_id), label_dir, remeshed_pin)
    results = []
    for case in audit:
        row = row_by_id[case["case"]]
        mesh = load_original_mesh(*verified_mesh_paths(train, row))
        direct = Simulation(str(train), case["case"])
        adapter = OriginalForceSimulation(
            mesh, float(direct.inlet_velocity), float(direct.angle_of_attack)
        )
        geometry = input_checks(mesh, adapter)
        forces = compare_forces(
            force_coefficients(adapter, reference=True),
            force_coefficients(direct, reference=True),
            np.asarray(case["stored"]),
        )
        results.append(
            case | {"mesh_files": row["files"], "inputs": geometry, "forces": forces}
        )
    passed = all(row["inputs"]["passed"] and row["forces"]["passed"] for row in results)
    with out.open("x") as stream:
        json.dump(
            {
                "schema": 1,
                "passed": passed,
                "training_stage_sha256": stage_digest,
                "label_source": remeshed_pin,
                "coefficient_names": ("CL", "CD"),
                "components": ("total", "pressure", "viscous"),
                "force_atol": {"adapter": ADAPTER_ATOL, "stored": STORED_ATOL},
                "git": git_state(),
                "versions": {
                    name: version(name) for name in ("airfrans", "vtk", "pyvista")
                },
                "cases": results,
            },
            stream,
            allow_nan=False,
            indent=2,
        )
        stream.write("\n")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_stage", type=Path)
    parser.add_argument("remeshed_labels", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not qualify(
        args.training_stage,
        args.remeshed_labels,
        json.loads(PIN.read_text()),
        json.loads(REMESHED_PIN.read_text()),
        args.out,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
