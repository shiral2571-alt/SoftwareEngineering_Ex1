# Experiment and Results

## Goals

1. Show that the demo produces correct, readable videos for triangles and trapezoids,
   including oblique planes that are not available as standard slice views.
2. Measure performance against the requirement (≥ 30 FPS on 500³, target 100 FPS).

## Setup

* Machine: Intel Core (family 6 model 186), Windows 11, Python 3.14, NumPy 2.5.
* Volume: `data/MRHead_1.nrrd` (256×256×130, int16) for the videos; random volumes for
  the benchmark.
* Commands:

```bash
python experiments/make_planes.py
python Ex0.py data/MRHead_1.nrrd data/planes_tumble.csv results/tumble.mp4 --fps 15
python experiments/make_previews.py
python experiments/benchmark.py
```

The other plane files in `data/` (sweep, rotate, triangle, sagittal5) can be rendered the
same way.

## Part 1: Video

| Video | Planes | Projection time | FPS |
|---|---|---|---|
| [tumble.mp4](../results/tumble.mp4) | 120 squares turning freely in 3D | 0.14 s | ~880 |

![Frames from tumble.mp4](../results/tumble_preview.png)

Observations:

* A 256×256 square through the head center, whose rotation axis itself keeps changing
  (two rotations at different speeds). The frames pass through sagittal (first frame),
  coronal, axial and many oblique slices that no standard slice view shows. The anatomy is
  continuous from frame to frame, which confirms that oblique planes and the row/column
  orientation are correct.
* Parts of the oblique squares fall outside the narrow (130-voxel) left-right axis. They are
  black (value 0), as required. These pixels all read index 0, which is always in the CPU
  cache, so this video runs faster than straight slices of the same size.
* Intensities: a global window [min, 99.5th percentile] keeps gray levels consistent across
  frames. Using the raw maximum made all frames dark, because a few very bright voxels
  stretched the scale.

## Part 2: Performance

From [results/benchmark.md](../results/benchmark.md) (each value is the mean over ≥ 1 s of
repeated projections, after one warm-up call), measured on an idle laptop (12 logical cores)
running on battery:

| Implementation / plane | Volume | Output | FPS |
|---|---|---|---:|
| Fast, oblique triangle | 128³ uint8 | 143×129 | 1,365 |
| Fast, oblique triangle | 256³ uint8 | 286×257 | 457 |
| Fast, oblique triangle | **500³ uint8** | 559×501 | **173.3** |
| Fast, oblique triangle | 750³ uint8 | 838×751 | 88.0 |
| Fast, oblique triangle | 500³ uint16 | 559×501 | 188.0 |
| Fast, oblique triangle | 500³ uint32 | 559×501 | 176.1 |
| Fast, axis-aligned square | 500³ uint8 | 499×500 | 163.8 |
| Fast, 1 thread (row blocks only), oblique triangle | 500³ uint8 | 559×501 | 142.6 |
| Fast, 1 thread (row blocks only), axis-aligned square | 500³ uint8 | 499×500 | 88.0 |
| Reference (loops), oblique triangle | 128³ uint8 | 143×129 | 24.0 |

Stability: five repeated 1-second runs of each 500³ case (uint8/16/32, oblique and
axis-aligned) gave **163–192 FPS**, so every run is above the 100 FPS target. Single numbers
depend on CPU load and power mode (with a busy browser and IDE, earlier versions lost up to
half of their speed).

Versions on the same 500³ oblique plane:

| Version | FPS |
|---|---:|
| First implementation (Python loops, accumulating `p += dp`) | 2–5 |
| First NumPy version (H×W×3 float64 array, boolean scatter) | ~29 |
| Gather, float32 per axis, int32 flat index, `take`, whole frame at once | 60–175 |
| The same with float64 coordinates (identical to the reference) | 61–93 |
| + row blocks of 32,768 pixels (one thread) | ~120–143 |
| **Final: + blocks on a pool of 4 threads** | **163–192** |

## Conclusions

* **The requirement and the target are met:** 163–192 FPS on 500³ (requirement ≥ 30,
  expected 100), in every repeated run. The requirement is tested automatically
  (`TestPerformance`).
* **Cache size matters more than arithmetic:** splitting the frame into blocks of rows, with
  no other change, made it ~1.3–1.5× faster, because the temporary arrays of a block (~1 MB)
  stay in the CPU cache between NumPy operations instead of going through RAM.
* **Threads work because of the gather design:** each block writes only its own rows, so blocks
  need no locks, and NumPy releases the GIL inside array operations. Four threads gave another
  ~1.3–2×; more threads did not help (memory bandwidth is shared).
* **Correctness was preferred over 15%:** float32 coordinates were faster but differed from the
  reference by one pixel on some long planes. float64 makes the two identical.
* **Run time depends on the number of output pixels, not on the volume:** MRHead frames
  (~256×300) run at 300–550 FPS.
* **Memory access matters at large sizes:** on one thread the axis-aligned square at `z = 250`
  is much *slower* than the oblique plane (88 vs 143 FPS), even though it has a similar number
  of pixels; with 4 threads the gap almost disappears. Along its rows only
  axis 0 changes, which is the slowest axis in C order, so consecutive samples are 250,000
  bytes apart and miss the CPU cache. Also, every pixel of the square is inside the volume,
  while about half of the triangle's canvas is empty.
* **Vectorizing alone was not enough:** the first NumPy version reached only 29 FPS because
  of large temporary arrays. Reducing memory traffic (gather instead of scatter, one axis at a
  time) gave the remaining speed-up.
* **The optimization is safe:** the fast and reference implementations are identical on the
  100 random planes of the oracle test, on long planes from far outside the volume, with
  1 or 4 threads and with tiny blocks, and on the real MRHead plane files.

## Possible improvements

* A compiled kernel (Numba `@njit(parallel=True)`) for the same loop: in a quick trial this
  gave ~200 FPS single-threaded and >2,000 FPS multi-threaded. It was not adopted, to avoid an
  extra dependency.
* Trilinear interpolation for smoother oblique slices (a new `Projector` strategy).
* Physical (mm) coordinates using the NRRD `space directions`.
