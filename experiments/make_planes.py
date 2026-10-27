"""Generate the plane CSV files used in the experiment (data/planes_*.csv).

All coordinates are voxel indices of data/MRHead_1.nrrd, shape (256, 256, 130).
In that file axis 0 runs anterior->posterior, axis 1 superior->inferior and
axis 2 is left-right, so a plane with constant axis-2 index is a sagittal slice.
"""
import os

import numpy as np

SHAPE = np.array([256, 256, 130])
CENTER = (SHAPE - 1) / 2
DOWN = np.array([0.0, 1.0, 0.0])          # image rows run top of head -> neck
HEIGHT = 256.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def horizontal(angle_deg):
    """Unit vector in the axis-0 / axis-2 (horizontal) plane."""
    t = np.radians(angle_deg)
    return np.array([np.cos(t), 0.0, np.sin(t)])


def rectangle(center, across, width):
    """Rectangle (special isosceles trapezoid) centered at `center`."""
    a1 = center - HEIGHT / 2 * DOWN - width / 2 * across
    a2 = center - HEIGHT / 2 * DOWN + width / 2 * across
    return [a1, a2, a1 + HEIGHT * DOWN, a2 + HEIGHT * DOWN]


def triangle(center, across, width):
    """Isosceles triangle with its apex at the top of the head."""
    apex = center - HEIGHT / 2 * DOWN
    return [apex, apex + HEIGHT * DOWN - width / 2 * across,
            apex + HEIGHT * DOWN + width / 2 * across]


def write_csv(name, planes):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w") as file:
        for points in planes:
            file.write(",".join(f"{v:.4f}" for p in points for v in p) + "\n")
    print(f"{path}: {len(planes)} planes")


def main():
    # 1. Sagittal sweep: one slice for every left-right index (130 frames)
    sweep = [rectangle(np.array([CENTER[0], CENTER[1], z]), horizontal(0), 256)
             for z in range(SHAPE[2])]
    write_csv("planes_sweep.csv", sweep)

    # 2. Rotation: a vertical plane through the head center rotating 180 degrees
    #    (sagittal -> oblique -> coronal -> oblique -> sagittal), 90 frames
    rotate = [rectangle(CENTER, horizontal(angle), 300) for angle in np.linspace(0, 180, 90)]
    write_csv("planes_rotate.csv", rotate)

    # 3. Triangles: an isosceles triangle hanging from the top of the head,
    #    rotating about the vertical axis, 60 frames
    fan = [triangle(CENTER, horizontal(angle), 300) for angle in np.linspace(0, 180, 60)]
    write_csv("planes_triangle.csv", fan)


if __name__ == "__main__":
    main()
