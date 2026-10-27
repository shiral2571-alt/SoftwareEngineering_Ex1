"""3D raster volume M[x][y][z] loaded from an NRRD file."""
from __future__ import annotations

import numpy as np

MAX_AXIS_SIZE = 2 ** 10


class Volume:
    """A validated 3D matrix plus its NRRD header.

    The data is stored C-contiguous so projectors can index it as a flat
    array without a per-frame copy (pynrrd returns Fortran-ordered arrays).
    """

    def __init__(self, data: np.ndarray, header: dict | None = None):
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"Volume must be 3D, got shape {data.shape}")
        if not all(1 <= n < MAX_AXIS_SIZE for n in data.shape):
            raise ValueError(f"Each axis size must be in [1, {MAX_AXIS_SIZE}), got {data.shape}")
        # The spec requires 8/16/32-bit unsigned voxels; any numeric type works.
        if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
            raise ValueError(f"Voxel type must be numeric, got {data.dtype}")
        self._data = np.ascontiguousarray(data)
        self._data.setflags(write=False)
        self._header = dict(header or {})

    @classmethod
    def from_nrrd(cls, path: str) -> "Volume":
        import nrrd  # imported lazily so the core logic does not depend on pynrrd
        data, header = nrrd.read(path)
        return cls(data, header)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def header(self) -> dict:
        return self._header

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def __repr__(self) -> str:
        return f"Volume(shape={self.shape}, dtype={self.dtype})"
