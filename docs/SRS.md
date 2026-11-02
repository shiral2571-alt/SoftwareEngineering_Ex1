# Software Requirements Specification: Plane Projection of a 3D Volume

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a program that computes the 2D image of a
3D raster volume along an arbitrary plane, and produces a video from a sequence of planes.

### 1.2 Scope
The system reads a volume (NRRD) and a list of planes (CSV), projects the volume onto each
plane, and writes an mp4 video. It is a command-line program plus a reusable Python package
(`projection`).

### 1.3 Definitions

| Term | Meaning |
|---|---|
| Volume `M` | 3D matrix `M[x][y][z]`, `0 ≤ x < sx`, `0 ≤ y < sy`, `0 ≤ z < sz` |
| Voxel | One entry of `M` |
| Plane | An isosceles triangle `(a, b, c)` or isosceles trapezoid `(a1, a2, b, c)` |
| Leg | The equal sides `a1→b` and `a2→c` (in a triangle, `a1 = a2 = a`) |
| `Dab` | Leg length, `dist(a1, b)` |
| `Dbc` | Base length, `dist(b, c)` |
| `E` | The output image (projection) |
| EPS | Geometric tolerance, `0.001` |
| FPS | Frames (projections) computed per second |

## 2. Overall description

### 2.1 Product perspective
A standalone tool, similar to the slice views of 3D Slicer, but for arbitrary oblique planes.

### 2.2 User classes
* **Student/grader:** runs the demo `python Ex0.py file.nrrd file.csv file.mp4`.
* **Developer:** uses the `projection` package from Python code.

### 2.3 Operating environment
Python ≥ 3.10 on Windows, Linux or macOS, with numpy, pynrrd, imageio and imageio-ffmpeg.

### 2.4 Assumptions
* A1. Plane coordinates are given in **voxel index space** of the volume, not physical (mm)
  space. The NRRD `space directions` and `origin` are kept in the header but not applied.
* A2. Voxels are sampled by **nearest neighbour** (rounding), not interpolation.
* A3. The volume fits in memory.

## 3. Functional requirements

### Input
| ID | Requirement |
|---|---|
| FR-1 | The system shall load a 3D volume from an NRRD file. |
| FR-2 | The system shall accept volumes with `1 ≤ sx, sy, sz < 2^10` and 8/16/32-bit integer voxels (other numeric types are accepted too). Other shapes are rejected. |
| FR-3 | The system shall read planes from a CSV file, one plane per row: 9 values (triangle `a,b,c`) or 12 values (trapezoid `a1,a2,b,c`). Blank lines are ignored and a non-numeric first row is treated as a header. |

### Validation
| ID | Requirement |
|---|---|
| FR-4 | Each point shall have exactly 3 finite coordinates. |
| FR-5 | Leg lengths shall satisfy `|dist(a1,b) − dist(a2,c)| < EPS`, and be at least 1 voxel. Legs and base shall be at most 4096 voxels. |
| FR-6 | The points shall not be collinear (they must span a plane). |
| FR-7 | For a trapezoid: `dist(a1,a2) ≤ dist(b,c)` (+EPS), and `a2` lies within EPS of the plane through `a1, b, c`. |
| FR-8 | An invalid plane shall be reported with its CSV line number and the reason, and the program shall exit with code 1 without writing a video. |

### Projection
| ID | Requirement |
|---|---|
| FR-9 | For each plane the system shall compute `E` of shape `(⌊Dab⌋, ⌊Dbc⌋ + 1)` using the algorithm in [ALGORITHM.md](ALGORITHM.md). |
| FR-10 | A sample point outside `M` shall produce the value 0. Planes that are partly or fully outside the volume are valid. |
| FR-11 | `E` shall have the same data type as `M` (no truncation of 16/32-bit values). |

### Output
| ID | Requirement |
|---|---|
| FR-12 | The system shall write an mp4 file with exactly one frame per CSV row, in CSV order. |
| FR-13 | Frames shall be converted to 8-bit gray with one global intensity window for the whole video (minimum value to the 99.5th percentile), so equal voxel values look the same in every frame. |
| FR-14 | Frames of different sizes shall be centered on a common canvas (padded with 0) whose sides are multiples of 16. |
| FR-15 | The program shall print the volume shape/type, the number of planes, the projection time and FPS. |

### Command-line interface
| ID | Requirement |
|---|---|
| FR-16 | Usage: `python Ex0.py <file.nrrd> <file.csv> <file.mp4> [--fps N]`. A missing argument prints a usage message. |
| FR-17 | A missing, unreadable or corrupt file, an output name not ending in `.mp4`, a non-positive `--fps` or lack of memory shall print `Error: ...` and exit with code 1 (no traceback). |

## 4. Non-functional requirements

| ID | Requirement | Verification |
|---|---|---|
| NFR-1 | **Performance:** ≥ 30 FPS for a 500³ volume (must); ≥ 100 FPS (target). | `tests/test_projector.py::TestPerformance`, `experiments/benchmark.py` |
| NFR-2 | **Correctness:** the optimized implementation shall produce output identical to the reference implementation. | Oracle test on 100 random planes |
| NFR-3 | **Robustness:** no crash on any input; every error is a clear message. | Validation and end-to-end tests |
| NFR-4 | **Memory:** per-frame memory is O(output pixels), independent of volume size; the volume is not copied per frame. | Design (C-contiguous volume, flat `take`) |
| NFR-5 | **Maintainability:** modular object-oriented design, each class with one responsibility; the core is testable without files. `projection/` package structure |
| NFR-6 | **Portability:** pure Python + NumPy, no compiled extensions of our own. | |

## 5. Design decisions resolving the under-specified algorithm

| # | Question | Decision | Reason |
|---|---|---|---|
| D1 | Non-integer `p`: which voxel? | Round each coordinate to nearest (`round`, ties to even) | Simple, unbiased, no blurring of the raw data |
| D2 | Order of "step" and "sample" in the outer loop | Step first, then sample (as written in the assignment) | Faithful to the given pseudo-code; documented in ALGORITHM.md |
| D3 | Output size `Dab × Dbc` with real-valued lengths | `⌊Dab⌋` rows × `⌊Dbc⌋ + 1` columns | Rows: loop `y < Dab`. Columns: `x1 ≤ d12` makes up to `⌊Dbc⌋ + 1` samples |
| D4 | `E[x][y]` vs image layout | Stored as `E[y, x]` (row = y) | Standard row-major image layout for display and video |
| D5 | Column offset `(Dbc/2) − (d12/2)` is real | Truncated to an integer: `⌊(Dbc − d12)/2⌋` | Centers each row; consistent across implementations |
| D6 | Value outside `M` | 0 | Specified by the assignment |
| D7 | Coordinate space | Voxel indices | Assignment defines `M[x][y][z]` by index (assumption A1) |
| D8 | Video intensity mapping | Global window [min, 99.5th percentile] | Consistent gray levels between frames, good contrast |
| D9 | Frames of different sizes | Center-pad to a common size (multiple of 16) | Video requires a fixed size; scaling would distort the geometry |
| D10 | Very small or very large planes | Reject legs < 1 voxel and sides > 4096 voxels | Legs < 1 give `⌊Dab⌋ = 0` rows (an empty image); 4096 is about twice the diagonal of the largest allowed volume, and larger planes would need gigabytes of memory |
