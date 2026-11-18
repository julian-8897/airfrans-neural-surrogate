import io
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

geometry = pytest.importorskip("airfoil_surrogate.cfd.geometry")


def _shape(scale=1.0):
    x = np.linspace(0, 1, 5)
    y = scale * np.array([0, 0.05, 0.06, 0.02, 0.0005])
    return np.column_stack((x, y)), np.column_stack((x, -y))


def _as_selig(label, upper, lower):
    contour = np.vstack((upper[::-1], lower[1:]))
    return (label + "\n" + "\n".join(f"{x:.8f} {y:.8f}" for x, y in contour)).encode()


def test_selig_and_lednicer_align_to_the_same_closed_profile():
    upper, lower = _shape()
    standard = _as_selig("Example", upper, lower) + b"\n# ignored comment\n\x1a"
    counts = "Example\n5. 5.\n\n" + "\n".join(
        f"{x} {y}" for x, y in np.vstack((upper, lower))
    )
    label, *parts = geometry.parse_coordinates(standard)
    assert label == "Example"
    profile = geometry.align_profile(*parts)
    lednicer = geometry.align_profile(*geometry.parse_coordinates(counts.encode())[1:])
    assert_allclose(profile.ordinate_samples, lednicer.ordinate_samples, rtol=0, atol=0)
    assert profile.trailing_gap == pytest.approx(0.001)
    assert profile.thickness == pytest.approx(0.12)
    assert profile.absolute_camber == pytest.approx(0.0)

    rotation = np.array([[0, 1], [-1, 0]])
    moved = geometry.align_profile(
        5 * upper @ rotation + [8, -2], 5 * lower @ rotation + [8, -2]
    )
    assert_allclose(moved.ordinate_samples, profile.ordinate_samples, atol=1e-14)
    contour = profile.contour()
    assert contour.shape == (4001, 2)
    assert_array_equal(contour[[0, -1]], [[1, 0], [1, 0]])
    assert_array_equal(contour[2000], [0, 0])


def test_trailing_edge_closure_leaves_upstream_interpolant_unchanged():
    from scipy.interpolate import PchipInterpolator

    upper, lower = _shape()
    profile = geometry.align_profile(upper, lower)
    single = PchipInterpolator(upper[:, 0], upper[:, 1])
    stations = np.array([0.97, 0.98, 0.99, 1.0])
    expected = single(stations) - np.array([0, 0, 0.5, 1.0]) * upper[-1, 1]
    assert_allclose(profile.upper(stations), expected, atol=1e-15)
    too_wide = upper.copy()
    too_wide[-1, 1] = 0.003
    with pytest.raises(ValueError, match="trailing-edge gap"):
        geometry.align_profile(too_wide, lower)
    crossed = upper.copy()
    crossed[2, 1] = -0.10
    with pytest.raises(ValueError, match="touch or cross"):
        geometry.align_profile(crossed, lower)


def test_uiuc_screen_uses_labels_airfrans_distance_and_hash_order():
    cohorts = pytest.importorskip("airfoil_surrogate.cfd.cohorts")
    naca = {"original-naca": geometry.align_profile(*_shape(1.4))}
    entries = {
        "copy-b.dat": (_as_selig("Example B", *_shape()), "Example B"),
        "copy-a.dat": (_as_selig("Example A", *_shape()), "Example A"),
        "modified.dat": (
            _as_selig("Modified section", *_shape()),
            "Derived from NACA 0012",
        ),
        "near.dat": (_as_selig("Example C", *_shape(1.4)), "Example C"),
        "unlisted.dat": (_as_selig("Example D", *_shape()), None),
    }
    index = "\n".join(
        f'<a href="coord/{name}">{name}</a> \\ {label} \\ <br>'
        for name, (_, label) in entries.items()
        if label is not None
    ).encode()
    source = io.BytesIO()
    with ZipFile(source, "w") as archive:
        for name, (raw, _) in entries.items():
            archive.writestr(f"coord_seligFmt/{name}", raw)
    with ZipFile(source) as archive:
        audited, retained = cohorts.screen_uiuc(index, archive, naca)
    expected = min(("copy-a.dat", "copy-b.dat"), key=cohorts.selection_key)
    assert [row["filename"] for row, _ in retained] == [expected]
    by_name = {row["filename"]: row for row in audited}
    other = (set(("copy-a.dat", "copy-b.dat")) - {expected}).pop()
    assert by_name[other]["duplicate_of"] == expected
    assert by_name["near.dat"]["nearest_naca_rms"] == pytest.approx(0, abs=1e-15)
    assert "NACA-derived" in by_name["modified.dat"]["reason"]
    assert "ambiguous identity" in by_name["unlisted.dat"]["reason"]
