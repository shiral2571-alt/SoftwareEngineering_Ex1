"""Plane projection of a 3D volume (Software Engineering, Ex0)."""
from __future__ import annotations

import numpy as np

from .plane import EPS, Plane
from .plane_reader import read_planes
from .projector import FastProjector, Projector, ReferenceProjector
from .video import VideoWriter, pad_to_size
from .volume import Volume

__all__ = ["EPS", "Plane", "Volume", "Projector", "FastProjector",
           "ReferenceProjector", "VideoWriter", "pad_to_size", "read_planes",
           "project_plane"]


def project_plane(M, a, b, c, a_second=None) -> np.ndarray:
    """Convenience function: project raw array M onto the plane (a, b, c)
    or the trapezoid (a, a_second, b, c) using the fast projector."""
    plane = Plane.triangle(a, b, c) if a_second is None else Plane.trapezoid(a, a_second, b, c)
    volume = M if isinstance(M, Volume) else Volume(M)
    return FastProjector().project(volume, plane)
