"""Synthetic meshes with known answers, shared by tests."""

import numpy as np

from airfoil_surrogate.data.remeshed import BOX


def grid_with_hole(hole: set[tuple[int, int]]):
    """Triangulated grid over BOX with the quads in ``hole`` removed."""
    xs = np.linspace(BOX[0], BOX[1], 7)
    ys = np.linspace(BOX[2], BOX[3], 7)
    pos = np.array([(x, y) for y in ys for x in xs])
    nid = lambda i, j: j * len(xs) + i  # noqa: E731
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            if (i, j) in hole:
                continue
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    return pos, np.array(tris), nid
