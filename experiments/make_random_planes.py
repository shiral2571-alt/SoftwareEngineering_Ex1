"""Write a CSV of random planes through MRHead (new planes on every run).

Usage:
    python experiments/make_random_planes.py [N] [--seed S] [--min-inside F]

Each plane is a random rectangle or isosceles triangle, at a random point near
the head center, with a random 3D orientation. Planes with less than a fraction
F (default 0.8) of their area inside the volume are discarded and drawn again,
so most of every frame shows the head. Output: data/planes_random.csv
"""
import argparse

import numpy as np

from make_planes import CENTER, DOWN, SHAPE, horizontal, write_csv

GRID = np.linspace(0, 1, 21)


def random_rotation(rng):
    """Uniformly random 3x3 rotation matrix (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_plane(rng):
    """Random rectangle [a1, a2, b, c] or triangle [a, b, c]."""
    R = random_rotation(rng)
    center = CENTER + rng.uniform(-20, 20, size=3)
    across, down = R @ horizontal(0), R @ DOWN
    width, height = rng.uniform(100, 220), rng.uniform(100, 220)
    top, bottom = center - height / 2 * down, center + height / 2 * down
    if rng.random() < 0.5:
        a1, a2 = top - width / 2 * across, top + width / 2 * across
        return [a1, a2, a1 + height * down, a2 + height * down]
    return [top, bottom - width / 2 * across, bottom + width / 2 * across]


def inside_fraction(points):
    """Fraction of a grid of sample points on the shape that lies in the volume."""
    s, t = np.meshgrid(GRID, GRID)
    s, t = s.reshape(-1, 1), t.reshape(-1, 1)
    if len(points) == 4:                      # rectangle: a1 + s*(a2-a1) + t*(b-a1)
        a1, a2, b, _ = points
        samples = a1 + s * (a2 - a1) + t * (b - a1)
    else:                                     # triangle: a + t*(b-a) + s*t*(c-b)
        a, b, c = points
        samples = a + t * (b - a) + s * t * (c - b)
    idx = np.round(samples)
    return np.mean(np.all((idx >= 0) & (idx < SHAPE), axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("n", nargs="?", type=int, default=60, help="number of planes")
    parser.add_argument("--seed", type=int, default=None, help="repeat the same planes")
    parser.add_argument("--min-inside", type=float, default=0.8,
                        help="minimal fraction of each plane inside the volume")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    planes = []
    while len(planes) < args.n:
        points = random_plane(rng)
        if inside_fraction(points) >= args.min_inside:
            planes.append(points)
    write_csv("planes_random.csv", planes)


if __name__ == "__main__":
    main()
