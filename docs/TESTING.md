# Testing Strategy

Run all tests (about 3 seconds):

```bash
python -m unittest -v
```

## Levels

| Level | File | What it checks |
|---|---|---|
| Unit: validation | `tests/test_plane.py` | Every SRS validation rule (FR-4 to FR-7), on both sides of each boundary: EPS just below/above, zero leg, collinear points, trapezoid top wider than base, non-coplanar trapezoid, wrong dimension, NaN, leg shorter than 1 voxel (rejected) and of exactly 1 voxel (one row), plane larger than 4096 voxels. Also immutability and defensive copying. |
| Unit: volume | `tests/test_projector.py::TestVolumeValidation` | FR-2: 2D array rejected, axis 1024 rejected and 1023 accepted, smallest volume 1×1×1, non-numeric voxels rejected. |
| Unit: known answers | `tests/test_projector.py::TestKnownAnswers` | Synthetic volume `M[x,y,z] = x + 100y + 10000z`: every voxel encodes its own coordinates, so the expected image can be written by hand. Also plane fully outside → all zeros, dtype and shape preserved for uint8/16/32, no truncation of 32-bit values, triangle rows widen towards the base. Runs on **both** projectors. |
| Unit: original tests | `tests/test_project_plane.py` | The first tests, written against the single-file version (constant/varying volumes, boundary crossing, invalid planes). Kept unchanged as regression tests. |
| Oracle (differential) | `tests/test_projector.py::TestFastMatchesReference` | 100 random triangles and trapezoids in random orientations, partly outside the volume: `FastProjector` must equal `ReferenceProjector` **pixel for pixel**. This is what makes the optimization trustworthy. Plus long planes (2500 voxels) starting far outside the volume, including two planes on which an earlier float32 version differed by one pixel. |
| Performance | `tests/test_projector.py::TestPerformance` | NFR-1: ≥ 30 FPS on a random 500³ uint8 volume with an oblique plane (after a warm-up call). |
| I/O | `tests/test_io_and_demo.py` | CSV: header, blank lines, 9/12 values, wrong count, invalid plane reported with its line number, empty file, semicolons/tabs/spaces/CRLF, trailing comma (an error, not a header). Video: frames of different sizes get one padded size, centered padding, global normalization, percentile clipping, constant frames (no division by zero), NaN voxels, invalid fps (0, negative, NaN, infinity, too small or too large). |
| End-to-end (demo) | `tests/test_io_and_demo.py::TestDemoEndToEnd` | Runs `Ex0.main` exactly like `python Ex0.py vol.nrrd planes.csv out.mp4` on a generated NRRD: exit code 0, the FPS line is printed, and **the mp4 has exactly one frame per CSV row**. Also missing file → exit code 1 with `Error`, invalid plane → exit code 1 with the line number. Robustness (NFR-3): corrupt NRRD, tiny plane, huge plane, output not `.mp4`, missing output folder and `--fps 0` all end with `Error: ...` and exit code 1, never a traceback. |

## Testing the demo manually

```bash
python Ex0.py data/MRHead_1.nrrd data/planes_sagittal5.csv out.mp4 --fps 2
```

Expected: `Volume: (256, 256, 130) int16`, `Planes: 5`, an FPS line, `Saved: out.mp4`, and a
2.5-second video with 5 different sagittal slices.

Error cases:

```bash
python Ex0.py missing.nrrd data/planes_sagittal5.csv out.mp4     # Error: ... No such file
python Ex0.py data/MRHead_1.nrrd                                 # usage message
```

## Why these tests

* The algorithm is under-specified, so the **reference implementation is the specification in
  code**. Known-answer tests check the reference against hand-computed results; the oracle
  test checks the fast version against the reference. Together they cover correctness without
  having to hand-compute large oblique images.
* Boundary tests follow the SRS rules one by one, so every requirement has a test.
* The performance requirement is tested automatically, so a later change that slows the
  code down below 30 FPS fails the build.
