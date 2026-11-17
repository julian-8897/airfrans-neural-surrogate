"""Examine model inputs and targets on the full_train split (exploratory).

Checks the wall mask and normals independently, summarises channel
statistics, and measures how target magnitudes scale with U_inf.

    uv run python scripts/examine_inputs.py [--out outputs/inputs_examination]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from airfoil_surrogate.data.features import (
    INPUT_NAMES,
    TARGET_NAMES,
    Normaliser,
    build_features,
)
from airfoil_surrogate.data.remeshed import iter_samples, load_split_indices

ROOT = Path(__file__).resolve().parents[1]


def node_gradient(triangles: np.ndarray, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted average of per-triangle linear gradients at each node."""
    p0, p1, p2 = (pos[triangles[:, k]] for k in range(3))
    f0, f1, f2 = (f[triangles[:, k]] for k in range(3))
    m = np.stack([p1 - p0, p2 - p0], axis=1)  # (M, 2, 2)
    rhs = np.stack([f1 - f0, f2 - f0], axis=1)
    grad = np.linalg.solve(m, rhs[..., None])[..., 0]
    area = 0.5 * np.abs(np.linalg.det(m))
    out = np.zeros_like(pos)
    weight = np.zeros(len(pos))
    for k in range(3):
        np.add.at(out, triangles[:, k], grad * area[:, None])
        np.add.at(weight, triangles[:, k], area)
    return out / weight[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/raw/airfrans_remeshed"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/inputs_examination")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_split_indices(args.data / "README.md")["full_train"]
    feats, cases = [], []
    mask_mismatch, normal_dot, speed_at_wall, flat_sdf = 0, [], [], 0
    first = None
    for s in iter_samples(args.data, train):
        first = first or s
        f = build_features(s)
        feats.append(f)
        # Wall mask from mesh topology vs no-slip nodes (|u| = 0).
        noslip = np.linalg.norm(s.fields[:, :2], axis=1) == 0
        mask_mismatch += int((noslip != f.surf).sum())
        speed_at_wall.append(
            np.linalg.norm(s.fields[f.surf, :2], axis=1) / s.inlet_velocity
        )
        # Normals should point up the signed-distance gradient (into fluid).
        # Skip nodes whose surrounding triangles all lie on the wall (flat sdf).
        g = node_gradient(s.triangles, s.pos, f.x[:, 4])[f.surf]
        g_norm = np.linalg.norm(g, axis=1)
        flat_sdf += int((g_norm == 0).sum())
        ok = g_norm > 0
        cos = np.einsum("ij,ij->i", g[ok] / g_norm[ok, None], f.x[f.surf][ok, 5:7])
        normal_dot.append(cos)
        u = s.inlet_velocity
        y = s.fields
        cases.append(
            (
                u,
                np.degrees(s.aoa),
                np.sqrt((y[:, 0] ** 2 + y[:, 1] ** 2).mean()),
                np.sqrt((y[:, 2] ** 2).mean()),
                np.sqrt((y[:, 3] ** 2).mean()),
                f.surf.sum(),
                len(f.x),
            )
        )
    cases = np.array(cases)
    normal_dot = np.concatenate(normal_dot)

    print(f"cases: {len(feats)}  nodes: {int(cases[:, 6].sum())}")
    print(f"wall mask vs no-slip nodes, mismatched nodes: {mask_mismatch}")
    wall_speed = np.concatenate(speed_at_wall)
    print(
        "|u|/U_inf on wall nodes, p50/p99/max: "
        f"{np.percentile(wall_speed, 50):.4f} / {np.percentile(wall_speed, 99):.4f}"
        f" / {wall_speed.max():.3f}"
    )
    print(f"wall nodes skipped in normal check (flat sdf): {flat_sdf}")
    print(
        "cos(normal, grad sdf) on wall: "
        f"min {normal_dot.min():.3f}  p1 {np.percentile(normal_dot, 1):.3f}  "
        f"median {np.median(normal_dot):.3f}"
    )
    print(f"wall-node fraction: {cases[:, 5].sum() / cases[:, 6].sum():.3%}")

    norm = Normaliser.fit(feats)
    x_all = np.concatenate([f.x for f in feats])
    y_all = np.concatenate([f.y for f in feats])
    print("\nchannel        mean        std        min        p1      p99       max")
    for names, data, mean, std in (
        (INPUT_NAMES, x_all, norm.mean_in, norm.std_in),
        (TARGET_NAMES, y_all, norm.mean_out, norm.std_out),
    ):
        for k, name in enumerate(names):
            c = data[:, k]
            p1, p99 = np.percentile(c, [1, 99])
            print(
                f"{name:9s} {mean[k]:10.4g} {std[k]:10.4g} {c.min():10.4g} "
                f"{p1:9.4g} {p99:9.4g} {c.max():9.4g}"
            )
    z = norm.targets(y_all)
    print("\nnormalised target |z| > 5 fraction:", np.round((np.abs(z) > 5).mean(0), 5))
    sk = ((z - z.mean(0)) ** 3).mean(0) / z.std(0) ** 3
    print("normalised target skewness:", np.round(sk, 2))

    # Power-law scaling of per-case RMS with U_inf (log-log slope).
    print("\nper-case RMS vs U_inf: slope (log-log), and spread of RMS / U_inf^slope")
    for col, name, expect in ((2, "|u|", 1), (3, "p", 2), (4, "nu_t", 1)):
        slope = np.polyfit(np.log(cases[:, 0]), np.log(cases[:, col]), 1)[0]
        raw_cv = cases[:, col].std() / cases[:, col].mean()
        nd = cases[:, col] / cases[:, 0] ** expect
        print(
            f"  {name:5s} slope {slope:5.2f} (dimensional analysis: {expect}); "
            f"CV raw {raw_cv:.2f} -> CV after / U^{expect}: {nd.std() / nd.mean():.2f}"
        )

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, name in enumerate(TARGET_NAMES):
        ax[0].hist(z[:, k], bins=200, histtype="step", density=True, label=name)
    ax[0].set(yscale="log", xlabel="normalised target", title="Target distributions")
    ax[0].legend()
    for col, name in ((2, "RMS |u|"), (3, "RMS p"), (4, "RMS nu_t")):
        ax[1].loglog(
            cases[:, 0], cases[:, col] / cases[:, col].mean(), ".", ms=3, label=name
        )
    ax[1].set(
        xlabel="U_inf (m/s)", ylabel="per-case RMS / mean", title="Scaling with U_inf"
    )
    ax[1].legend()
    ax[2].hist(normal_dot, bins=100)
    ax[2].set(xlabel="cos(normal, grad sdf)", yscale="log", title="Wall normal check")
    fig.tight_layout()
    fig.savefig(args.out / "inputs_summary.png", dpi=130)

    f0 = feats[0]
    tri = mtri.Triangulation(*f0.pos.T, first.triangles)  # mesh triangles keep the hole
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sc = ax[0].tripcolor(tri, f0.y[:, 2], shading="gouraud", cmap="RdBu_r")
    fig.colorbar(sc, ax=ax[0], label="p (m²/s²)")
    ax[0].set(
        aspect="equal",
        title=f"case {first.index}: pressure",
        xlim=(-0.5, 1.5),
        ylim=(-0.5, 0.5),
    )
    w = f0.surf
    ax[1].plot(*f0.pos[w].T, ".", ms=1, color="k")
    ax[1].quiver(*f0.pos[w][::8].T, *f0.x[w][::8, 5:7].T, scale=40, width=0.002)
    ax[1].set(
        aspect="equal", title="wall nodes and normals (every 8th)", xlim=(-0.1, 1.1)
    )
    fig.tight_layout()
    fig.savefig(args.out / "case_fields_normals.png", dpi=130)
    print(f"\nfigures written to {args.out}")


if __name__ == "__main__":
    main()
