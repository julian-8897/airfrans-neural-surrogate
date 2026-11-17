"""Per-node model inputs and targets, following the AirfRANS convention.

Inputs (7 channels): x, y, U_inf cos(alpha), U_inf sin(alpha), signed distance
(positive in the fluid), and the wall normal (n_x, n_y), which is zero away
from the wall. Targets (4 channels): u_x, u_y, p, nu_t. Wall normals point out
of the airfoil into the fluid.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airfoil_surrogate.data.remeshed import RemeshedSample, wall_edges

INPUT_NAMES = ("x", "y", "u_inf_x", "u_inf_y", "sdf", "n_x", "n_y")
TARGET_NAMES = ("u_x", "u_y", "p", "nu_t")


@dataclass(frozen=True)
class NodeFeatures:
    pos: np.ndarray  # (N, 2) raw coordinates
    x: np.ndarray  # (N, 7) inputs, INPUT_NAMES order
    y: np.ndarray  # (N, 4) targets, TARGET_NAMES order
    surf: np.ndarray  # (N,) bool, wall nodes


def wall_normals(triangles: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Unit normals at wall nodes, pointing into the fluid; zero elsewhere.

    Each wall edge gets the perpendicular oriented towards the opposite vertex
    of its (fluid) triangle. Node normals are the length-weighted sum of the
    two adjacent edge normals, then normalised.
    """
    edges, opposite = wall_edges(triangles, pos)
    a, b = pos[edges[:, 0]], pos[edges[:, 1]]
    t = b - a
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)  # |n| = edge length
    into_fluid = np.einsum("ij,ij->i", n, pos[opposite] - a) > 0
    n[~into_fluid] *= -1
    normals = np.zeros_like(pos)
    np.add.at(normals, edges[:, 0], n)
    np.add.at(normals, edges[:, 1], n)
    wall = np.unique(edges)
    normals[wall] /= np.linalg.norm(normals[wall], axis=1, keepdims=True)
    return normals


def build_features(sample: RemeshedSample) -> NodeFeatures:
    pos = sample.pos
    n_nodes = len(pos)
    u_inf = sample.inlet_velocity * np.array([np.cos(sample.aoa), np.sin(sample.aoa)])
    normals = wall_normals(sample.triangles, pos)
    surf = np.zeros(n_nodes, dtype=bool)
    surf[np.unique(wall_edges(sample.triangles, pos)[0])] = True
    x = np.concatenate(
        [
            pos,
            np.broadcast_to(u_inf, (n_nodes, 2)),
            -sample.implicit_distance[:, None],
            normals,
        ],
        axis=1,
    )
    return NodeFeatures(pos=pos, x=x, y=sample.fields.copy(), surf=surf)


@dataclass(frozen=True)
class Normaliser:
    """Per-channel affine scaling fitted on training nodes only."""

    mean_in: np.ndarray
    std_in: np.ndarray
    mean_out: np.ndarray
    std_out: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, features: Iterable[NodeFeatures]) -> Normaliser:
        # Pooled over all nodes of all training cases, in float64.
        n, s_in, ss_in, s_out, ss_out = 0, 0.0, 0.0, 0.0, 0.0
        for f in features:
            n += len(f.x)
            s_in = s_in + f.x.sum(0)
            ss_in = ss_in + (f.x**2).sum(0)
            s_out = s_out + f.y.sum(0)
            ss_out = ss_out + (f.y**2).sum(0)
        mean_in, mean_out = s_in / n, s_out / n
        std_in = np.sqrt(np.maximum(ss_in / n - mean_in**2, 0.0))
        std_out = np.sqrt(np.maximum(ss_out / n - mean_out**2, 0.0))
        return cls(mean_in, std_in, mean_out, std_out)

    def inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_in) / (self.std_in + self.eps)

    def targets(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean_out) / (self.std_out + self.eps)

    def inverse_targets(self, y: np.ndarray) -> np.ndarray:
        return y * (self.std_out + self.eps) + self.mean_out

    def save(self, path: Path) -> None:
        data = {
            k: getattr(self, k).tolist()
            for k in ("mean_in", "std_in", "mean_out", "std_out")
        }
        data["eps"] = self.eps
        data["input_names"], data["target_names"] = INPUT_NAMES, TARGET_NAMES
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Normaliser:
        d = json.loads(Path(path).read_text())
        arrays = {
            k: np.asarray(d[k]) for k in ("mean_in", "std_in", "mean_out", "std_out")
        }
        return cls(**arrays, eps=d["eps"])
