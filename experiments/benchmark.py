"""Measure projection speed (frames per second) and write results/benchmark.md.

Run:  python experiments/benchmark.py
"""
import os
import platform
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from projection import FastProjector, Plane, ReferenceProjector, Volume  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark.md")


def measure_fps(projector, volume, plane, min_seconds=1.0):
    projector.project(volume, plane)  # warm-up
    frames, start = 0, perf_counter()
    while perf_counter() - start < min_seconds:
        projector.project(volume, plane)
        frames += 1
    return frames / (perf_counter() - start)


def oblique_triangle(n):
    """A tilted isosceles triangle crossing an n^3 volume (~n x n pixels)."""
    return Plane.triangle([n / 2, 0, 0], [0, n / 2, 0.866 * n], [n, n / 2, 0.866 * n])


def random_volume(n, dtype):
    info = np.iinfo(dtype)
    return Volume(np.random.default_rng(0).integers(0, info.max, (n, n, n), dtype=dtype,
                                                    endpoint=True))


def main():
    fast = FastProjector()
    rows = []

    def add(name, volume, plane, projector=fast):
        fps = measure_fps(projector, volume, plane)
        shape = "x".join(map(str, plane.output_shape))
        rows.append((name, f"{volume.shape[0]}^3 {volume.dtype}", shape, f"{fps:,.1f}"))
        print(rows[-1])

    print("Volume size (uint8, oblique triangle):")
    for n in (128, 256, 500, 750):
        add("Fast, oblique triangle", random_volume(n, np.uint8), oblique_triangle(n))

    print("Voxel type (500^3, oblique triangle):")
    for dtype in (np.uint16, np.uint32):
        add("Fast, oblique triangle", random_volume(500, dtype), oblique_triangle(500))

    # the original benchmark: 500^3 constant volume, axis-aligned square at z=250
    constant = Volume(np.full((500, 500, 500), 7, dtype=np.uint8))
    square = Plane.trapezoid([0, 0, 250], [499, 0, 250], [0, 499, 250], [499, 499, 250])
    add("Fast, axis-aligned square", constant, square)

    print("Reference (loop) implementation, for comparison:")
    add("Reference, oblique triangle", random_volume(128, np.uint8), oblique_triangle(128),
        ReferenceProjector())

    with open(RESULTS, "w") as file:
        file.write("# Benchmark results\n\n")
        file.write(f"Machine: {platform.processor() or platform.machine()}, "
                   f"Python {platform.python_version()}, NumPy {np.__version__}\n\n")
        file.write("| Implementation / plane | Volume | Output (rows x cols) | FPS |\n")
        file.write("|---|---|---|---:|\n")
        for row in rows:
            file.write("| " + " | ".join(row) + " |\n")
    print(f"Saved {RESULTS}")


if __name__ == "__main__":
    main()
