"""Projection strategies: given a Volume and a Plane, produce the 2D image E.

Both strategies implement exactly the same algorithm (see docs/ALGORITHM.md):

    for y in 0 .. H-1:
        a1 = A1 + (y+1) * v(a1,b)          # step first, then sample
        a2 = A2 + (y+1) * v(a2,c)
        d12 = |a2 - a1|,  dp = (a2 - a1) / d12
        for x1 in 0 .. floor(d12):
            p = a1 + x1 * dp
            E[y, floor((Dbc - d12) / 2) + x1] = M[round(p)]  (0 if outside M)

ReferenceProjector is the readable, loop-based version (the test oracle).
FastProjector is the vectorized NumPy version used in production.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .plane import Plane
from .volume import Volume


class Projector(ABC):
    """Strategy interface for projecting a volume onto a plane."""

    @abstractmethod
    def project(self, volume: Volume, plane: Plane) -> np.ndarray:
        """Return E with shape plane.output_shape and the volume's dtype."""


class ReferenceProjector(Projector):
    """Direct, pixel-by-pixel translation of the algorithm. Slow but obvious."""

    def project(self, volume: Volume, plane: Plane) -> np.ndarray:
        M = volume.data
        height, width = plane.output_shape
        d_bc = plane.base_length
        v_ab, v_ac = plane.leg_directions
        E = np.zeros((height, width), dtype=M.dtype)

        for y in range(height):
            a1 = plane.a1 + (y + 1) * v_ab
            a2 = plane.a2 + (y + 1) * v_ac
            d12 = float(np.linalg.norm(a2 - a1))
            if d12 == 0:
                continue
            dp = (a2 - a1) / d12
            offset = int((d_bc - d12) / 2)

            for x1 in range(int(d12) + 1):
                x = offset + x1
                if x >= width:
                    break
                p = a1 + x1 * dp
                i, j, k = (int(round(float(coord))) for coord in p)
                if 0 <= i < M.shape[0] and 0 <= j < M.shape[1] and 0 <= k < M.shape[2]:
                    E[y, x] = M[i, j, k]
        return E


class FastProjector(Projector):
    """Vectorized implementation: all pixels of a frame are computed at once.

    Instead of scattering every sample p to its column x, it gathers: for each
    output pixel (y, x) it computes the x1 = x - offset(y) that lands there.
    Coordinates are computed one axis at a time into a reused buffer,
    combined into a flat int32 index (sx*sy*sz < 2^30), and sampled with a
    single `take` call. Coordinates use float64 with the same operations as
    the reference (a1 + x1*dp, round half to even), so both give identical
    images; float32 was ~15% faster but rounded a few pixels differently.
    """

    def project(self, volume: Volume, plane: Plane) -> np.ndarray:
        M = volume.data
        height, width = plane.output_shape
        d_bc = plane.base_length
        v_ab, v_ac = plane.leg_directions

        steps = np.arange(1, height + 1)[:, None]
        A1 = plane.a1 + steps * v_ab
        A2 = plane.a2 + steps * v_ac

        d12 = np.linalg.norm(A2 - A1, axis=1)
        dp = (A2 - A1) / np.where(d12 > 0, d12, 1)[:, None]

        offset = ((d_bc - d12) / 2).astype(np.int32)
        x1 = np.arange(width, dtype=np.int32)[None, :] - offset[:, None]
        valid = (x1 >= 0) & (x1 <= d12[:, None].astype(np.int32)) & (d12[:, None] > 0)

        x1f = x1.astype(np.float64)
        flat = np.zeros((height, width), dtype=np.int32)
        coord = np.empty((height, width), dtype=np.float64)
        for axis, size in enumerate(M.shape):
            np.multiply(x1f, dp[:, axis][:, None], out=coord)
            coord += A1[:, axis][:, None]
            np.rint(coord, out=coord)
            valid &= (coord >= 0) & (coord < size)
            flat *= size
            flat += coord.astype(np.int32)

        # invalid pixels read index 0 (always safe) and are then zeroed
        flat *= valid
        E = M.ravel().take(flat)
        E *= valid
        return E
