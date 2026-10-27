# Design

## Classes and responsibilities

| Module | Class / function | Responsibility | Depends on |
|---|---|---|---|
| `plane.py` | `Plane` | Holds the 4 points of a triangle/trapezoid; validates them; exposes geometry (`side_length`, `base_length`, `leg_directions`, `output_shape`) | numpy |
| `volume.py` | `Volume` | Holds a validated, read-only, C-contiguous 3D array and its NRRD header; `from_nrrd(path)` | numpy, pynrrd (lazy import) |
| `projector.py` | `Projector` (ABC) | Interface: `project(volume, plane) -> E` | Plane, Volume |
| | `ReferenceProjector` | Direct loop translation of the algorithm (specification and test oracle) | |
| | `FastProjector` | Vectorized NumPy implementation (production) | |
| `plane_reader.py` | `read_planes(csv)` | Parses CSV rows into `Plane` objects, with line-numbered errors | Plane |
| `video.py` | `VideoWriter` | Intensity normalization, padding to a common size, mp4 encoding | imageio (lazy import) |
| `Ex0.py` | `main(argv)` | CLI: parses arguments, wires the components, reports errors and timing | everything above |

```
        Ex0.main
     /    |     \          \
read_planes  Volume  FastProjector  VideoWriter
     |                 |  (Projector)
   Plane  <------------+
```

## Design choices

**Strategy pattern for the projector.**
`ReferenceProjector` and `FastProjector` are interchangeable behind the `Projector`
interface. The reference version is easy to check against the assignment's pseudo-code, and
the fast version is checked against it automatically. A future implementation (Numba, GPU,
trilinear interpolation) can be added without touching the CLI, the I/O or the tests.

**`Plane` is an immutable value object that validates itself.**
All construction paths (`triangle`, `trapezoid`, `from_values`) go through `__init__`, which
validates. The points are copied and made read-only, so a `Plane` cannot become invalid after
creation and no other code has to re-check it. Triangle and trapezoid are one class (a
triangle is a trapezoid with `a1 = a2`), so the projector has a single code path.

**`Volume` owns the memory layout.**
pynrrd returns Fortran-ordered arrays. `Volume` converts once to C order and makes the data
read-only. This is what lets `FastProjector` use `M.ravel()` as a free view instead of copying
125 MB on every frame.

**I/O at the edges.**
Reading NRRD/CSV and writing mp4 happen only in `Volume.from_nrrd`, `read_planes`,
`VideoWriter.write` and `Ex0.py`. `Plane`, `Projector` and `VideoWriter.prepare` are pure
and are unit-tested without files. pynrrd and imageio are imported lazily, so the core
package works without them.

**Errors are values, not crashes.**
Library code raises `ValueError` with a precise message, `read_planes` adds the CSV line
number, and `Ex0.main` turns `ValueError`/`OSError` into `Error: ...` with exit code 1.
`main(argv)` returns an exit code instead of calling `sys.exit`, so the end-to-end tests can
call it directly.

**Backward-compatible helper.**
`projection.project_plane(M, a, b, c, a_second=None)` keeps the original function signature,
so the first tests written for the single-file version still run unchanged.
