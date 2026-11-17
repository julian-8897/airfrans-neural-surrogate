import numpy as np
import pytest
from meshes import grid_with_hole

from airfoil_surrogate.data.features import (
    NodeFeatures,
    Normaliser,
    build_features,
    wall_normals,
)
from airfoil_surrogate.data.remeshed import RemeshedSample

HOLE = {(2, 2), (3, 2), (2, 3), (3, 3)}  # 2x2 quads around grid node (3, 3)


def test_wall_normals_point_away_from_hole_into_fluid():
    pos, tris, nid = grid_with_hole(HOLE)
    normals = wall_normals(tris, pos)
    centre = pos[nid(3, 3)]
    wall = np.flatnonzero(np.linalg.norm(normals, axis=1))
    assert len(wall) == 8
    np.testing.assert_allclose(np.linalg.norm(normals[wall], axis=1), 1.0)
    # Rectangular cells (hx = 1, hy = 0.5). Mid-side nodes get the axis normal;
    # a corner gets the length-weighted sum of its two edge normals, which is
    # proportional to (sign(dx) * hy, sign(dy) * hx), pointing away from the hole.
    hx, hy = 1.0, 0.5
    d = pos[wall] - centre
    expected = np.stack([np.sign(d[:, 0]) * hy, np.sign(d[:, 1]) * hx], axis=1)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(normals[wall], expected, atol=1e-12)


def make_sample(aoa_deg: float = 10.0, u: float = 50.0) -> RemeshedSample:
    pos, tris, _ = grid_with_hole(HOLE)
    rng = np.random.default_rng(0)
    n = len(pos)
    return RemeshedSample(
        index=0,
        pos=pos,
        triangles=tris,
        fields=rng.normal(size=(n, 4)),
        implicit_distance=-rng.random(n),
        cell_zone=np.zeros(len(tris), dtype=np.int8),
        inlet_velocity=u,
        aoa=np.radians(aoa_deg),
        cl=0.0,
        cd=0.0,
    )


def test_build_features_channels():
    s = make_sample(aoa_deg=10.0, u=50.0)
    f = build_features(s)
    assert f.x.shape == (len(s.pos), 7) and f.y.shape == (len(s.pos), 4)
    np.testing.assert_array_equal(f.x[:, :2], s.pos)
    np.testing.assert_allclose(f.x[:, 2], 50 * np.cos(np.radians(10)))
    np.testing.assert_allclose(f.x[:, 3], 50 * np.sin(np.radians(10)))
    np.testing.assert_array_equal(f.x[:, 4], -s.implicit_distance)
    assert f.surf.sum() == 8
    assert np.all(f.x[~f.surf, 5:7] == 0)


def test_normaliser_standardises_pooled_training_nodes():
    rng = np.random.default_rng(1)
    feats = [
        NodeFeatures(
            pos=np.zeros((n, 2)),
            x=rng.normal(3.0, 2.0, size=(n, 7)),
            y=rng.normal(-1.0, 5.0, size=(n, 4)),
            surf=np.zeros(n, bool),
        )
        for n in (50, 300, 1000)
    ]
    norm = Normaliser.fit(feats)
    zx = norm.inputs(np.concatenate([f.x for f in feats]))
    zy = norm.targets(np.concatenate([f.y for f in feats]))
    np.testing.assert_allclose(zx.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(zx.std(0), 1, atol=1e-8)
    np.testing.assert_allclose(zy.std(0), 1, atol=1e-8)
    np.testing.assert_allclose(
        norm.inverse_targets(zy), np.concatenate([f.y for f in feats])
    )


def test_normaliser_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    f = NodeFeatures(
        np.zeros((10, 2)),
        rng.normal(size=(10, 7)),
        rng.normal(size=(10, 4)),
        np.zeros(10, bool),
    )
    norm = Normaliser.fit([f])
    norm.save(tmp_path / "norm.json")
    loaded = Normaliser.load(tmp_path / "norm.json")
    for k in ("mean_in", "std_in", "mean_out", "std_out"):
        np.testing.assert_array_equal(getattr(loaded, k), getattr(norm, k))
    assert loaded.eps == pytest.approx(norm.eps)
