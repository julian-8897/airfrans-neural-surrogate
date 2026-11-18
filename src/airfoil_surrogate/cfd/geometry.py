"""Read and align published airfoil coordinates for the matched-flow study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

CHECK_X = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 201)))[1:-1]
MESH_X = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 2001)))


@dataclass(frozen=True)
class Profile:
    upper: Callable[[np.ndarray], np.ndarray]
    lower: Callable[[np.ndarray], np.ndarray]
    ordinate_samples: np.ndarray
    trailing_gap: float

    @property
    def thickness(self) -> float:
        return float(np.max(self.ordinate_samples[0] - self.ordinate_samples[1]))

    @property
    def absolute_camber(self) -> float:
        return float(np.max(np.abs(self.ordinate_samples.mean(axis=0))))

    def contour(self) -> np.ndarray:
        """Return upper TE to LE, followed by the lower surface back to the TE."""
        upper = np.column_stack((MESH_X, self.upper(MESH_X)))
        lower = np.column_stack((MESH_X, self.lower(MESH_X)))
        return np.vstack((upper[::-1], lower[1:]))


def split_contour(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a Selig-order contour at the point farthest from its trailing edge."""
    trailing = (points[0] + points[-1]) / 2
    leading = int(np.argmax(((points - trailing) ** 2).sum(axis=1)))
    if not 0 < leading < len(points) - 1:
        raise ValueError("leading edge is not inside the contour")
    return points[: leading + 1][::-1], points[leading:]


def _close_tail(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    interpolant = PchipInterpolator(points[:, 0], points[:, 1], extrapolate=False)
    end_x, end_y = points[-1]

    def sample(x: np.ndarray) -> np.ndarray:
        fraction = np.clip((x - 0.98) / 0.02, 0, 1)
        blend = fraction**2 * (3 - 2 * fraction)
        # Evaluate the original spline at the corresponding chord position.
        # At x <= 0.98 both coordinates are untouched.
        raw_x = np.minimum(x + blend * (end_x - 1), end_x)
        return np.where(x == 1, 0, interpolant(raw_x) - blend * end_y)

    return sample


def parse_coordinates(raw: bytes) -> tuple[str, np.ndarray, np.ndarray]:
    """Read Selig or Lednicer coordinates, stripping embedded comment lines."""
    text = raw.decode("latin-1").replace("\x1a", "")
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("empty coordinate file")

    def pair(line: str) -> tuple[float, float]:
        first, second = line.split()
        return float(first), float(second)

    label = ""
    try:
        pair(lines[0])
    except ValueError:
        label, lines = lines[0], lines[1:]
    coordinates = np.array([pair(line) for line in lines])
    if len(coordinates) < 5 or not np.isfinite(coordinates).all():
        raise ValueError("expected at least five finite coordinate pairs")

    first, second = coordinates[0]
    if first >= 3 and second >= 3 and first.is_integer() and second.is_integer():
        if int(first + second) != len(coordinates) - 1:
            raise ValueError("Lednicer surface counts do not match the coordinate file")
        split = 1 + int(first)
        return label, coordinates[1:split], coordinates[split:]
    return label, *split_contour(coordinates)


def align_profile(first: np.ndarray, second: np.ndarray) -> Profile:
    """Align to the unit chord and close the trailing edge over the final 2%."""
    if not np.array_equal(first[0], second[0]):
        raise ValueError("surface branches do not share a leading edge")
    leading = first[0]
    trailing = (first[-1] + second[-1]) / 2
    chord = np.linalg.norm(trailing - leading)
    if not np.isfinite(chord) or chord == 0:
        raise ValueError("invalid chord length")
    gap = float(np.linalg.norm(first[-1] - second[-1]) / chord)
    if gap > 0.002:
        raise ValueError("trailing-edge gap exceeds 0.002 chord")
    tangent = (trailing - leading) / chord
    basis = np.column_stack((tangent, [-tangent[1], tangent[0]]))

    curves = []
    for branch in (first, second):
        points = (branch - leading) @ basis / chord
        # Remove only adjacent, identical coordinates; retaining them would
        # introduce a zero-length spline segment.
        points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
        points[0] = [0, 0]
        if len(points) < 3 or not np.all(np.diff(points[:, 0]) > 0):
            raise ValueError("surface is not single-valued along the chord")
        curve = _close_tail(points)
        curves.append((curve(CHECK_X), curve))
    (upper_y, upper), (lower_y, lower) = sorted(
        curves, key=lambda item: item[0].mean(), reverse=True
    )
    if np.any(upper_y <= lower_y):
        raise ValueError("surface branches touch or cross inside the chord")
    return Profile(upper, lower, np.stack((upper_y, lower_y)), gap)


def passes_shape_screen(profile: Profile) -> bool:
    return 0.07 <= profile.thickness <= 0.18 and profile.absolute_camber <= 0.05
