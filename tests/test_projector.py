import time
import unittest

import numpy as np

from projection import FastProjector, Plane, ReferenceProjector, Volume


def random_plane(rng, low=-10, high=50):
    """A random valid triangle or trapezoid, possibly sticking out of the volume."""
    origin = rng.uniform(low, high, 3)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    w = rng.normal(size=3)
    w -= w.dot(u) * u
    w /= np.linalg.norm(w)
    length = rng.uniform(3, 60)
    half = rng.uniform(0.5, 1.5) * length
    far = origin + length * u
    if rng.random() < 0.5:
        return Plane.triangle(origin, far - half * w, far + half * w)
    top = rng.uniform(0, 1) * half
    return Plane.trapezoid(origin - top * w, origin + top * w, far - half * w, far + half * w)


class TestKnownAnswers(unittest.TestCase):
    """Synthetic volumes where the correct image can be computed by hand."""

    def setUp(self):
        x, y, z = np.indices((20, 30, 40))
        self.coordinates = Volume((x + 100 * y + 10000 * z).astype(np.int32))
        self.projectors = [ReferenceProjector(), FastProjector()]

    def test_axis_aligned_rectangle_reads_exact_voxels(self):
        # a1=(2,5,7) -> b=(2,15,7); every row y samples M[2+x1, 5+y+1, 7]
        plane = Plane.trapezoid([2, 5, 7], [12, 5, 7], [2, 15, 7], [12, 15, 7])
        expected = np.array([[(2 + x) + 100 * (6 + y) + 10000 * 7 for x in range(11)]
                             for y in range(10)])
        for projector in self.projectors:
            np.testing.assert_array_equal(projector.project(self.coordinates, plane), expected)

    def test_slice_at_constant_z_is_constant_in_z_component(self):
        plane = Plane.trapezoid([0, 0, 13], [19, 0, 13], [0, 29, 13], [19, 29, 13])
        for projector in self.projectors:
            E = projector.project(self.coordinates, plane)
            self.assertTrue(np.all(E // 10000 == 13))

    def test_plane_fully_outside_is_black(self):
        plane = Plane.triangle([100, 100, 100], [110, 105, 100], [110, 95, 100])
        for projector in self.projectors:
            self.assertFalse(projector.project(self.coordinates, plane).any())

    def test_output_shape_and_dtype(self):
        plane = Plane.triangle([0, 0, 0], [3, 4, 0], [-3, 4, 0])
        for dtype in (np.uint8, np.uint16, np.uint32):
            volume = Volume(np.ones((8, 8, 8), dtype=dtype))
            for projector in self.projectors:
                E = projector.project(volume, plane)
                self.assertEqual(E.shape, (5, 7))
                self.assertEqual(E.dtype, dtype)

    def test_uint32_values_are_not_truncated(self):
        volume = Volume(np.full((4, 4, 4), 4_000_000_000, dtype=np.uint32))
        plane = Plane.trapezoid([0, 0, 1], [3, 0, 1], [0, 3, 1], [3, 3, 1])
        for projector in self.projectors:
            self.assertEqual(projector.project(volume, plane).max(), 4_000_000_000)

    def test_triangle_rows_widen_towards_base(self):
        volume = Volume(np.ones((40, 40, 40), dtype=np.uint8))
        plane = Plane.triangle([20, 0, 5], [0, 30, 5], [40, 30, 5])
        for projector in self.projectors:
            filled = np.count_nonzero(projector.project(volume, plane), axis=1)
            self.assertTrue(np.all(np.diff(filled) >= 0))


class TestVolumeValidation(unittest.TestCase):

    def test_not_3d(self):
        with self.assertRaises(ValueError):
            Volume(np.zeros((4, 4), np.uint8))

    def test_axis_of_1024_is_rejected_and_1023_accepted(self):
        with self.assertRaises(ValueError):
            Volume(np.zeros((1024, 1, 1), np.uint8))
        Volume(np.zeros((1023, 1, 1), np.uint8))

    def test_smallest_volume_1x1x1(self):
        volume = Volume(np.full((1, 1, 1), 7, np.uint8))
        plane = Plane.triangle([0, 0, 0], [0, 2, -1], [0, 2, 1])
        for projector in (ReferenceProjector(), FastProjector()):
            E = projector.project(volume, plane)
            self.assertEqual(E[0, 0], 0)          # first sample is (0, 1, 0): outside
            self.assertEqual(E.shape, plane.output_shape)

    def test_non_numeric_voxels(self):
        with self.assertRaises(ValueError):
            Volume(np.zeros((2, 2, 2), dtype=bool))


class TestFastMatchesReference(unittest.TestCase):
    """Oracle test: the optimized projector must equal the readable one."""

    def test_random_planes(self):
        rng = np.random.default_rng(2024)
        volume = Volume(rng.integers(0, 256, (40, 50, 30), dtype=np.uint8))
        reference, fast = ReferenceProjector(), FastProjector()
        for _ in range(100):
            plane = random_plane(rng)
            with self.subTest(plane=plane):
                np.testing.assert_array_equal(fast.project(volume, plane),
                                              reference.project(volume, plane))


class TestPerformance(unittest.TestCase):
    """Requirement NFR-1: at least 30 frames per second on a 500^3 volume."""

    REQUIRED_FPS = 30

    def test_500_cubed_meets_required_fps(self):
        rng = np.random.default_rng(0)
        volume = Volume(rng.integers(0, 256, (500, 500, 500), dtype=np.uint8))
        plane = Plane.triangle([250, 0, 0], [0, 250, 433], [500, 250, 433])
        projector = FastProjector()
        projector.project(volume, plane)  # warm-up

        repeats = 30
        start = time.perf_counter()
        for _ in range(repeats):
            projector.project(volume, plane)
        fps = repeats / (time.perf_counter() - start)
        self.assertGreaterEqual(fps, self.REQUIRED_FPS, f"only {fps:.1f} fps")


if __name__ == "__main__":
    unittest.main()
