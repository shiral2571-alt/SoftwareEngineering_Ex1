import unittest

import numpy as np

from projection import EPS, Plane


class TestPlaneConstruction(unittest.TestCase):

    def test_triangle_geometry(self):
        plane = Plane.triangle([0, 0, 0], [3, 4, 0], [-3, 4, 0])
        self.assertTrue(plane.is_triangle)
        self.assertAlmostEqual(plane.side_length, 5)
        self.assertAlmostEqual(plane.base_length, 6)
        self.assertEqual(plane.output_shape, (5, 7))

    def test_trapezoid_geometry(self):
        plane = Plane.trapezoid([1, 0, 0], [3, 0, 0], [0, 4, 0], [4, 4, 0])
        self.assertFalse(plane.is_triangle)
        self.assertEqual(plane.output_shape, (int(np.hypot(1, 4)), 5))

    def test_from_values_9_and_12(self):
        self.assertTrue(Plane.from_values([0, 0, 0, 3, 4, 0, -3, 4, 0]).is_triangle)
        self.assertFalse(Plane.from_values([1, 0, 0, 3, 0, 0, 0, 4, 0, 4, 4, 0]).is_triangle)

    def test_from_values_wrong_count(self):
        with self.assertRaises(ValueError):
            Plane.from_values([0] * 10)

    def test_points_are_immutable(self):
        plane = Plane.triangle([0, 0, 0], [3, 4, 0], [-3, 4, 0])
        with self.assertRaises(ValueError):
            plane.b[0] = 99

    def test_input_lists_are_copied(self):
        a = [0.0, 0.0, 0.0]
        plane = Plane.triangle(a, [3, 4, 0], [-3, 4, 0])
        a[0] = 100
        self.assertEqual(plane.a1[0], 0)


class TestPlaneValidation(unittest.TestCase):

    def test_side_difference_just_below_eps_is_accepted(self):
        d = 10 + EPS / 2
        Plane.triangle([0, 0, 0], [10, 0, 0], [0, d, 0])

    def test_side_difference_above_eps_is_rejected(self):
        with self.assertRaises(ValueError):
            Plane.triangle([0, 0, 0], [10, 0, 0], [0, 10 + 2 * EPS, 0])

    def test_zero_length_side(self):
        with self.assertRaises(ValueError):
            Plane.triangle([1, 1, 1], [1, 1, 1], [1, 1, 1])

    def test_collinear_points(self):
        with self.assertRaises(ValueError):
            Plane.triangle([0, 0, 0], [5, 0, 0], [-5, 0, 0])

    def test_trapezoid_top_wider_than_base(self):
        with self.assertRaises(ValueError):
            Plane.trapezoid([-5, 0, 0], [5, 0, 0], [-1, 4, 0], [1, 4, 0])

    def test_trapezoid_not_coplanar(self):
        angle = np.radians(30)
        with self.assertRaises(ValueError):
            Plane.trapezoid([200, 0, 0], [300, 0, 0], [0, 300, 0],
                            [500, 300 * np.cos(angle), 300 * np.sin(angle)])

    def test_point_with_wrong_dimension(self):
        with self.assertRaises(ValueError):
            Plane.triangle([0, 0], [3, 4, 0], [-3, 4, 0])

    def test_non_finite_coordinates(self):
        with self.assertRaises(ValueError):
            Plane.triangle([0, 0, np.nan], [3, 4, 0], [-3, 4, 0])


if __name__ == "__main__":
    unittest.main()
