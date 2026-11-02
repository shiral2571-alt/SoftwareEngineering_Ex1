"""The cutting plane: an isosceles triangle (3 points) or trapezoid (4 points)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 0.001
MIN_SIDE = 1.0      # shorter legs give int(Dab) == 0 rows: an empty image
MAX_SIDE = 4096.0   # about 2x the diagonal of the largest allowed volume (1023^3)


class Plane:
    """An immutable, always-valid cutting plane.

    The plane is stored in its general (trapezoid) form: two legs a1->b and
    a2->c of equal length. A triangle is the special case a1 == a2 == a.
    Every constructor path goes through validation, so a Plane object that
    exists is guaranteed to satisfy the assignment's preconditions.
    """

    __slots__ = ("_a1", "_a2", "_b", "_c")

    def __init__(self, a1: Sequence[float], a2: Sequence[float],
                 b: Sequence[float], c: Sequence[float]):
        a1, a2, b, c = (self._as_point(p) for p in (a1, a2, b, c))
        self._validate(a1, a2, b, c)
        for point in (a1, a2, b, c):
            point.setflags(write=False)
        self._a1, self._a2, self._b, self._c = a1, a2, b, c

    # ---- factories ------------------------------------------------------

    @classmethod
    def triangle(cls, a, b, c) -> "Plane":
        """Isosceles triangle (a, b, c) with dist(a,b) == dist(a,c)."""
        return cls(a, a, b, c)

    @classmethod
    def trapezoid(cls, a1, a2, b, c) -> "Plane":
        """Isosceles trapezoid with legs a1->b and a2->c."""
        return cls(a1, a2, b, c)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "Plane":
        """Build from a flat list: 9 numbers (a,b,c) or 12 numbers (a1,a2,b,c)."""
        if len(values) == 9:
            return cls.triangle(values[0:3], values[3:6], values[6:9])
        if len(values) == 12:
            return cls.trapezoid(values[0:3], values[3:6], values[6:9], values[9:12])
        raise ValueError(f"A plane needs 9 or 12 numbers, got {len(values)}")

    # ---- geometry -------------------------------------------------------

    @property
    def a1(self) -> np.ndarray:
        return self._a1

    @property
    def a2(self) -> np.ndarray:
        return self._a2

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def is_triangle(self) -> bool:
        return bool(np.array_equal(self._a1, self._a2))

    @property
    def side_length(self) -> float:
        """Dab: length of the legs (dist(a1,b) == dist(a2,c))."""
        return float(np.linalg.norm(self._b - self._a1))

    @property
    def base_length(self) -> float:
        """Dbc: length of the far base (b, c)."""
        return float(np.linalg.norm(self._c - self._b))

    @property
    def leg_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors v(a1,b) and v(a2,c)."""
        return ((self._b - self._a1) / np.linalg.norm(self._b - self._a1),
                (self._c - self._a2) / np.linalg.norm(self._c - self._a2))

    @property
    def output_shape(self) -> tuple[int, int]:
        """(rows, cols) of the projected image: (int(Dab), int(Dbc) + 1)."""
        return int(self.side_length), int(self.base_length) + 1

    def __repr__(self) -> str:
        if self.is_triangle:
            return f"Plane.triangle(a={self._a1.tolist()}, b={self._b.tolist()}, c={self._c.tolist()})"
        return (f"Plane.trapezoid(a1={self._a1.tolist()}, a2={self._a2.tolist()}, "
                f"b={self._b.tolist()}, c={self._c.tolist()})")

    # ---- validation -----------------------------------------------------

    @staticmethod
    def _as_point(point) -> np.ndarray:
        arr = np.array(point, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"A point must have exactly 3 coordinates, got {point!r}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Point coordinates must be finite numbers, got {point!r}")
        return arr

    @staticmethod
    def _validate(a1, a2, b, c) -> None:
        d_ab = np.linalg.norm(b - a1)
        d_ac = np.linalg.norm(c - a2)
        if d_ab == 0 or d_ac == 0:
            raise ValueError("Side lengths must be greater than zero")
        if abs(d_ab - d_ac) >= EPS:
            raise ValueError(f"Side lengths must differ by less than {EPS} "
                             f"(got {d_ab:.4f} and {d_ac:.4f})")
        if d_ab < MIN_SIDE:
            raise ValueError(f"Side length must be at least {MIN_SIDE:g} voxel "
                             f"(got {d_ab:.4f}), otherwise the image has no rows")
        d_bc = np.linalg.norm(c - b)
        if max(d_ab, d_bc) > MAX_SIDE:
            raise ValueError(f"Plane is too large: sides must be at most {MAX_SIDE:g} voxels "
                             f"(got {d_ab:.1f} and {d_bc:.1f})")

        normal = np.cross(b - a1, c - a1)
        normal_length = np.linalg.norm(normal)
        if normal_length < EPS:
            raise ValueError("The points are collinear and do not define a plane")

        if not np.array_equal(a1, a2):
            if np.linalg.norm(a2 - a1) > np.linalg.norm(c - b) + EPS:
                raise ValueError("dist(a1,a2) must be <= dist(b,c)")
            if abs(np.dot(normal, a2 - a1)) / normal_length >= EPS:
                raise ValueError("All 4 points must lie on the same plane")
