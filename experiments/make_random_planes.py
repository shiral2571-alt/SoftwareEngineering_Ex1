"""Write a CSV of random planes through MRHead (new planes on every run).

Usage:
    python experiments/make_random_planes.py [N] [--seed S]

Each plane is a random rectangle or isosceles triangle, at a random point near
the head center, with a random 3D orientation. Output: data/planes_random.csv
"""
import argparse

import numpy as np

from make_planes import CENTER, DOWN, horizontal, rectangle, write_csv


def random_rotation(rng):
    """Uniformly random 3x3 rotation matrix (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_plane(rng):
    R = random_rotation(rng)
    center = CENTER + rng.uniform(-30, 30, size=3)
    across, down = R @ horizontal(0), R @ DOWN
    width = rng.uniform(150, 300)
    if rng.random() < 0.5:
        return rectangle(center, across, width, down=down)
    apex = center - 128 * down
    base = center + 128 * down
    return [apex, base - width / 2 * across, base + width / 2 * across]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("n", nargs="?", type=int, default=60, help="number of planes")
    parser.add_argument("--seed", type=int, default=None, help="repeat the same planes")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    write_csv("planes_random.csv", [random_plane(rng) for _ in range(args.n)])


if __name__ == "__main__":
    main()
